# Created by MacBook Pro at 30.04.25


from typing import Callable, List, Optional
import torch
# simple struct to hold a predicate definition
class PredicateDef:
    def __init__(
        self,
        name: str,
        arity: int,
        is_neural: bool = False,
        score_fn: Optional[Callable] = None,
        threshold: Optional[float] = None,
    ):
        """
        name      : predicate name (e.g. 'shape' or 'proximity')
        arity     : number of arguments
        is_neural : if True, we call score_fn(...) to get a continuous score
        score_fn  : function(args...) -> Tensor of shape (batch,) giving [0..1] scores
        threshold : if set, a score > threshold is treated as True
        """
        self.name = name
        self.arity = arity
        self.is_neural = is_neural
        self.score_fn = score_fn
        self.threshold = threshold

    def evaluate(self, *args, **kwargs) -> torch.Tensor:
        """
        If symbolic: args are precomputed booleans (Tensor[batch]), we just return them.
        If neural: calls score_fn to get scores, then thresholds if requested.
        """
        if not self.is_neural:
            # assume args[0] is already a BoolTensor
            return args[0].bool()
        else:
            scores: torch.Tensor = self.score_fn(*args, **kwargs)  # (batch,)
            if self.threshold is None:
                return scores
            return (scores > self.threshold).bool()


class LogicLanguage:
    def __init__(self):
        self.predicates: List[PredicateDef] = []
        self._add_symbolic()
        self._add_neural()

    def _add_symbolic(self):
        # group-level symbolic features
        self.predicates += [
            PredicateDef("shape",    2),  # shape(group_id, shape_id)
            PredicateDef("color",    2),  # color(group_id, color_id)
            PredicateDef("closure_position", 1),  # closure_position(group_id)
            PredicateDef("closure_feature",  1),
            PredicateDef("symmetry", 2),
        ]

    def _add_neural(self):
        # e.g. proximity: continuous scorer over two group embeddings
        from mbg.scorer.context_proximity_scorer import ContextProximityScorer
        from mbg.scorer import scorer_config
        # instantiate your pretrained scorer
        prox_model = ContextProximityScorer()
        prox_model.load_state_dict(torch.load(scorer_config.PROXIMITY_MODEL))
        prox_model.eval()

        # wrap it: group_repr_i, group_repr_j -> score in [0,1]
        def proximity_score(i_repr: torch.Tensor, j_repr: torch.Tensor) -> torch.Tensor:
            # i_repr, j_repr: (batch, embed_dim)
            return torch.sigmoid(prox_model(i_repr, j_repr))

        # add as a neural predicate, threshold at 0.5 by default
        self.predicates.append(
            PredicateDef(
                name="proximity",
                arity=2,
                is_neural=True,
                score_fn=proximity_score,
                threshold=0.5,
            )
        )

    def get(self, name:str) -> PredicateDef:
        for p in self.predicates:
            if p.name == name:
                return p
        raise KeyError(f"Predicate {name} not in language")