import numpy as np
import torch
from torch.utils.data import Dataset


class StudentSequenceDataset(Dataset):
    def __init__(
        self,
        sequences,
        question_embeddings,
        question_concepts,
        question_features=None,
        question_option_counts=None,
        max_seq_len=200,
        K=1175,
        return_indices=False,
    ):
        self.max_seq_len = max_seq_len
        self.K = K
        self.question_features = question_features
        self.return_indices = return_indices

        # 1. Build question ID to matrix index mapping
        # Index 0 is reserved for padding question (all zeros)
        q_ids_unique = list(question_embeddings.keys())
        self.q_id_to_idx = {q_id: idx + 1 for idx, q_id in enumerate(q_ids_unique)}
        num_questions = len(q_ids_unique)
        
        # 2. Determine max concepts per step across questions
        self.max_c = 1
        for q_id, concepts in question_concepts.items():
            active_concepts = [cid for cid in concepts if 0 <= cid < K]
            if len(active_concepts) > self.max_c:
                self.max_c = len(active_concepts)
        self.max_c = max(self.max_c, 1)
        print(f"Max concepts per question: {self.max_c}")
        
        # 3. Pre-build question attribute matrices
        print("Pre-building question metadata matrices...")
        self.question_embeddings_matrix = np.zeros((num_questions + 1, 1024), dtype=np.float32)
        self.question_concepts_matrix = np.full((num_questions + 1, self.max_c), -1, dtype=np.int64)
        self.question_concepts_matrix[0, 0] = 0  # Prevents Softmax NaN at padding steps where question index is 0
        self.question_g_priors = np.zeros(num_questions + 1, dtype=np.float32)
        
        if question_features is not None:
            self.question_features_matrix = np.zeros((num_questions + 1, 22), dtype=np.float32)
            
        for q_id, idx in self.q_id_to_idx.items():
            # Embeddings
            emb = question_embeddings.get(q_id)
            if emb is not None:
                self.question_embeddings_matrix[idx] = emb
                
            # Concepts
            db_concepts = question_concepts.get(q_id, [])
            active_concepts = [cid for cid in db_concepts if 0 <= cid < K]
            if not active_concepts:
                active_concepts = [0]
            self.question_concepts_matrix[idx, :len(active_concepts)] = active_concepts
            
            # Guess priors
            opt_cnt = question_option_counts.get(q_id, 0) if question_option_counts else 0
            g_prior = 1.0 / opt_cnt if opt_cnt >= 2 else 0.25
            self.question_g_priors[idx] = g_prior
            
            # Tabular features
            if question_features is not None:
                feat = question_features.get(q_id)
                if feat is not None:
                    self.question_features_matrix[idx] = feat

        # Pre-convert numpy matrices to PyTorch Tensors to avoid on-the-fly conversion overhead in __getitem__
        self.question_embeddings_matrix = torch.from_numpy(self.question_embeddings_matrix)
        self.question_concepts_matrix = torch.from_numpy(self.question_concepts_matrix)
        self.question_g_priors = torch.from_numpy(self.question_g_priors)
        if question_features is not None:
            self.question_features_matrix = torch.from_numpy(self.question_features_matrix)

        # 4. Pre-parse sequences to list of integer indices (O(1) RAM friendly)
        print("Pre-parsing student sequences...")
        self.processed = []
        for session in sequences:
            if not session.questions or not session.responses:
                continue
            
            q_ids_seq = [q.strip() for q in session.questions.split(",") if q.strip()]
            r_strings = [r.strip() for r in session.responses.split(",") if r.strip()]
            
            seq_len = min(len(q_ids_seq), len(r_strings), max_seq_len)
            # 1. Bỏ qua chuỗi tương tác quá ngắn (seq_len < 3)
            if seq_len < 3:
                continue
            
            q_ids_seq = q_ids_seq[:seq_len]
            r_strings = r_strings[:seq_len]
            
            if session.response_time:
                raw_times = [t.strip() for t in session.response_time.split(",") if t.strip()]
            else:
                raw_times = []

            # 2. Bỏ qua session bấm bừa liên tục (average response time < 3.0s)
            if len(raw_times) > 0:
                try:
                    times_numeric = []
                    for t in raw_times:
                        try:
                            val = float(t.strip())
                            if val > 0:
                                times_numeric.append(val)
                        except ValueError:
                            pass
                    if len(times_numeric) > 0 and np.mean(times_numeric) < 3.0:
                        continue
                except Exception:
                    pass

            # 4.1 Map question IDs to matrix indices
            q_indices = np.zeros(self.max_seq_len, dtype=np.int64)
            for t in range(seq_len):
                q_indices[t] = self.q_id_to_idx.get(q_ids_seq[t], 0)

            # 4.2 Parse responses
            r_arr = np.zeros(self.max_seq_len, dtype=np.float32)
            for t in range(seq_len):
                try:
                    r_arr[t] = float(r_strings[t])
                except ValueError:
                    r_arr[t] = 0.0

            # 4.3 Parse response times
            time_arr = np.zeros(self.max_seq_len, dtype=np.float32)
            for t in range(seq_len):
                try:
                    diff = float(raw_times[t]) if t < len(raw_times) else 30.0
                except (ValueError, IndexError):
                    diff = 30.0
                if diff < 1.0 or diff > 300.0:
                    diff = 30.0
                time_arr[t] = diff

            # 4.4 Padding mask
            mask_arr = np.zeros(self.max_seq_len, dtype=np.bool_)
            mask_arr[:seq_len] = True

            self.processed.append(
                {
                    "q_indices": q_indices,
                    "r_arr": r_arr,
                    "time_arr": time_arr,
                    "mask_arr": mask_arr,
                }
            )
            
        print(f"Pre-parsed and optimized {len(self.processed)} sequences successfully.")

    def __len__(self):
        return len(self.processed)

    def __getitem__(self, index):
        item = self.processed[index]
        q_indices = item["q_indices"]
        r = item["r_arr"]
        T_time = item["time_arr"]
        padding_mask = item["mask_arr"]

        if self.return_indices:
            return (
                torch.from_numpy(q_indices),
                torch.from_numpy(r),
                torch.from_numpy(T_time),
                torch.from_numpy(padding_mask),
            )
        
        # O(1) Vectorized lookup directly on CPU Tensors (compatible fallback)
        x = self.question_embeddings_matrix[q_indices]
        concept_indices = self.question_concepts_matrix[q_indices]
        g_priors = self.question_g_priors[q_indices]
        
        if self.question_features is not None:
            x_feat = self.question_features_matrix[q_indices]
            return (
                x,
                x_feat,
                torch.from_numpy(r),
                torch.from_numpy(T_time),
                concept_indices,
                torch.from_numpy(padding_mask),
                g_priors,
            )
        else:
            return (
                x,
                torch.from_numpy(r),
                torch.from_numpy(T_time),
                concept_indices,
                torch.from_numpy(padding_mask),
                g_priors,
            )
