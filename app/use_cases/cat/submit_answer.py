from typing import Dict, Any, Tuple, Optional
import numpy as np

from app.domain.entities.cat_entities import AdaptiveSession
from app.domain.interfaces.cat_ports import CATModelPort

class SubmitAnswerUseCase:
    """
    Use case to process a student's answer in an adaptive testing session.
    Updates ability estimate, calculates standard error, checks stopping conditions,
    and runs the 4PL-based item selection algorithm to pick the next question.
    """
    
    def __init__(self, cat_model: CATModelPort):
        self.cat_model = cat_model

    def execute(
        self,
        session: AdaptiveSession,
        questions_bank: Dict[str, Dict[str, Any]],
        question_id: str,
        is_correct: int,
        response_time: float,
        info_history: list  # Pass in from controller/UI session state to track Fisher Info values
    ) -> Tuple[AdaptiveSession, Optional[Dict[str, Any]], float]:
        """
        Processes a student response submission.
        Returns:
            session: Updated AdaptiveSession entity.
            next_question: Next optimal question dict, or None if stopping condition met.
            current_se: The estimated Standard Error after this step.
        """
        if session.is_completed:
            return session, None, session.se_history[-1] if session.se_history else 1.0

        # Verify the question exists in the bank
        if question_id not in questions_bank:
            raise ValueError(f"Question ID {question_id} not found in questions bank.")

        # 1. Update session interaction lists
        session.responses.append(is_correct)
        session.response_times.append(response_time)
        t = len(session.responses)  # Number of questions completed so far

        # Fetch metadata tables
        question_concepts = {qid: q["concept_ids"] for qid, q in questions_bank.items()}
        question_embeddings = {qid: q["embedding"] for qid, q in questions_bank.items()}
        question_features = {qid: q["features"] for qid, q in questions_bank.items()} if session.model_type == "optimized" else None
        question_option_counts = {qid: q["option_count"] for qid, q in questions_bank.items()}

        # 2. Run model forward pass to estimate new student mastery
        theta_prev, mastery_scores = self.cat_model.estimate_mastery(
            question_ids=session.questions_asked[:t],
            responses=session.responses,
            response_times=session.response_times,
            question_concepts=question_concepts,
            question_embeddings=question_embeddings,
            question_features=question_features,
            question_option_counts=question_option_counts
        )
        
        session.current_theta_state = theta_prev
        session.mastery_history.append(mastery_scores)

        # 3. Calculate Fisher Information for the current question based on student's ability BEFORE this step
        # This aligns with the adaptive testing principle where SE updates are calculated step-by-step
        # We need the predictions for this question at step t-1 (which was stored or we calculate here)
        prev_theta = session.current_theta_state
        # For simplicity, we calculate the Fisher Information for the question just answered
        # using the student's NEW mastery. In practice, both approaches are highly correlated,
        # but using the updated theta gives the most post-hoc accurate SE.
        curr_pred = self.cat_model.predict_candidates(
            theta_prev=theta_prev,
            candidate_ids=[question_id],
            question_concepts=question_concepts,
            question_embeddings=question_embeddings,
            question_features=question_features,
            question_option_counts=question_option_counts
        ).get(question_id)

        if curr_pred:
            g_q = curr_pred["g"]
            s_q = curr_pred["s"]
            delta_q = curr_pred["delta"]
            p_q = curr_pred["p_correct"]
            k_q = 1.0 / (1.0 + np.exp(-delta_q))  # sigmoid(delta)
            
            # Info(q) = [(1 - s - g) * K * (1 - K)]^2 / [P * (1 - P)]
            numerator = ((1.0 - s_q - g_q) * k_q * (1.0 - k_q)) ** 2
            denominator = max(1e-6, p_q * (1.0 - p_q))
            info_val = numerator / denominator
        else:
            info_val = 0.1  # Fallback minimum information
            
        info_history.append(info_val)

        # 4. Compute Cumulative Standard Error: SE_t = 1 / sqrt(sum(Info))
        sum_info = sum(info_history)
        current_se = 1.0 / np.sqrt(max(1e-4, sum_info))
        session.se_history.append(current_se)

        # 5. Check if Stopping Criteria is satisfied
        if session.check_stopping_condition(current_se):
            return session, None, current_se

        # 6. Run Item Selection to find next question
        # Filter out questions already asked
        asked_set = set(session.questions_asked)
        candidate_ids = [qid for qid in questions_bank.keys() if qid not in asked_set]
        
        if not candidate_ids:
            # Bank exhausted
            session.is_completed = True
            return session, None, current_se

        # Batch predict parameters for all candidate questions
        candidate_preds = self.cat_model.predict_candidates(
            theta_prev=theta_prev,
            candidate_ids=candidate_ids,
            question_concepts=question_concepts,
            question_embeddings=question_embeddings,
            question_features=question_features,
            question_option_counts=question_option_counts
        )

        # Compute Score(j) for all candidates
        lambda_t = session.get_lambda(t)
        
        # Determine content coverage frequency f_k
        # f_k = 1 if concept k has NOT been asked yet in the session, 0 otherwise.
        asked_concepts = set()
        for qid in session.questions_asked[:t]:
            asked_concepts.update(question_concepts.get(qid, []))
            
        best_candidate_qid = None
        best_score = -float("inf")

        for qid in candidate_ids:
            pred = candidate_preds.get(qid)
            if not pred:
                continue
                
            g_j = pred["g"]
            s_j = pred["s"]
            delta_j = pred["delta"]
            p_j = pred["p_correct"]
            k_j = 1.0 / (1.0 + np.exp(-delta_j))  # sigmoid(delta)
            
            # Calculate Info(j)
            num_j = ((1.0 - s_j - g_j) * k_j * (1.0 - k_j)) ** 2
            den_j = max(1e-6, p_j * (1.0 - p_j))
            info_j = num_j / den_j
            
            # Calculate Bonus(j)
            concepts_j = question_concepts.get(qid, [])
            active_concepts_j = [cid for cid in concepts_j if 0 <= cid < session.current_theta_state.shape[1] - 1]
            
            # Sum of concepts in question j that haven't been asked yet
            bonus_j = sum(1.0 for cid in active_concepts_j if cid not in asked_concepts)
            
            # Combine Info and Bonus using selection method
            if session.selection_method == "addition":
                # Linear addition: Score(j) = (1 - lambda) * Info(j) + lambda * Bonus(j)
                score_j = (1.0 - lambda_t) * info_j + lambda_t * bonus_j
            else:
                # Multiplication: Score(j) = Info(j) * Bonus(j)
                # We add a small epsilon to bonus_j to avoid completely neutralizing Info(j) 
                # if all concepts in the question have already been asked.
                score_j = info_j * max(0.01, bonus_j)

            if score_j > best_score:
                best_score = score_j
                best_candidate_qid = qid

        # Fallback if no candidate selected
        if not best_candidate_qid:
            best_candidate_qid = candidate_ids[0]

        # Add chosen question to history
        session.questions_asked.append(best_candidate_qid)
        next_question = questions_bank[best_candidate_qid]

        return session, next_question, current_se
