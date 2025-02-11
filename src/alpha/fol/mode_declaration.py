# Created by X at 25.06.24
from .logic import *
from src import bk


class ModeDeclaration(object):
    """from https://www.cs.ox.ac.uk/activities/programinduction/Aleph/aleph.html
    p(ModeType, ModeType,...)

    Here are some examples of how they appear in a file:

    :- mode(1,mem(+number,+list)).
    :- mode(1,dec(+integer,-integer)).
    :- mode(1,mult(+integer,+integer,-integer)).
    :- mode(1,plus(+integer,+integer,-integer)).
    :- mode(1,(+integer)=(#integer)).
    :- mode(*,has_car(+train,-car)).
    Each ModeType is either (a) simple; or (b) structured.
    A simple ModeType is one of:
    (a) +T specifying that when a literal with predicate symbol p appears in a
    hypothesised clause, the corresponding argument should be an "input" variable of type T;
    (b) -T specifying that the argument is an "output" variable of type T; or
    (c) #T specifying that it should be a constant of type T.
    All the examples above have simple modetypes.
    A structured ModeType is of the form f(..) where f is a function symbol,
    each argument of which is either a simple or structured ModeType.
    Here is an example containing a structured ModeType:


    To make this more clear, here is an example for the mode declarations for
    the grandfather task from
     above::- modeh(1, grandfather(+human, +human)).:-
      modeb(*, parent(-human, +human)).:-
       modeb(*, male(+human)).
       The  first  mode  states  that  the  head  of  the  rule
        (and  therefore  the  target predicate) will be the atom grandfather.
         Its parameters have to be of the type human.
          The  +  annotation  says  that  the  rule  head  needs  two  variables.
            The second mode declaration states the parent atom and declares again
             that the parameters have to be of type human.
              Here,  the + at the second parameter tells, that the system is only allowed to
              introduce the atom parent in the clause if it already contains a variable of type human.
               That the first attribute introduces a new variable into the clause.
    The  modes  consist  of  a  recall n that  states  how  many  versions  of  the
    literal are allowed in a rule and an atom with place-markers that state the literal to-gether
    with annotations on input- and output-variables as well as constants (see[Mug95]).
    Args:
        recall (int): The recall number i.e. how many times the declaration can be instanciated
        pred (Predicate): The predicate.
        mode_terms (ModeTerm): Terms for mode declarations.
    """

    def __init__(self, mode_type, recall, pred, mode_terms, ordered=True):
        self.mode_type = mode_type  # head or body
        self.recall = recall
        self.pred = pred
        self.mode_terms = mode_terms
        self.ordered = ordered

    def __str__(self):
        s = 'mode_' + self.mode_type + '('
        if self.mode_terms is None:
            raise ValueError
        for mt in self.mode_terms:
            s += str(mt)
            s += ','
        s = s[0:-1]
        s += ')'
        return s

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.__str__())


def get_pred_by_name(preds, pred_name):
    pred = [pred for pred in preds if pred.name == pred_name]
    if not len(pred) == 1:
        print("")
    return pred[0]


def get_invented_pred_by_name(preds, invented_pred_name):
    invented_pred = [inv_p for inv_p in preds if inv_p.name == invented_pred_name]
    if not len(invented_pred) == 1:
        raise ValueError('Too many or less match in ' + invented_pred_name)
    return invented_pred[0]


def get_mode_data(dtypes):
    mode_terms = []
    for dt in dtypes:
        term = ModeTerm(dt.sign, dt)
        mode_terms.append(term)
    return mode_terms


def get_mode_declarations_bk(preds):
    modeb_list = []
    for pred in preds:
        if pred.name in bk.mode_excluded_preds:
            continue
        pred_name = pred.name
        dtypes = pred.dtypes
        recall = bk.mode_recall
        mode_terms = get_mode_data(dtypes)
        modeb_list.append(
            ModeDeclaration('body', recall, get_pred_by_name(preds, pred_name), mode_terms, ordered=True))
    return modeb_list


def get_pi_mode_declarations(inv_preds, obj_num):
    p_object = ModeTerm('+', DataType('output_group'))
    pi_mode_declarations = []
    for pi_index, pi in enumerate(inv_preds):
        pi_str = pi.name
        objects = [p_object] * pi.arity
        mode_declarations = ModeDeclaration('body', obj_num, get_pred_by_name(inv_preds, pi_str), objects,
                                            ordered=False)
        pi_mode_declarations.append(mode_declarations)
    return pi_mode_declarations


def get_mode_declarations(predicates):
    basic_mode_declarations = get_mode_declarations_bk(predicates)
    # pi_model_declarations = get_pi_mode_declarations(inv_predicates, e)
    return basic_mode_declarations  # + pi_model_declarations
