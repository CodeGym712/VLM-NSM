(define (domain stacking)
  (:requirements :strips :typing)
  (:types block table - location)

  (:predicates
    (on ?block - block ?location - location)
    (clear ?location - location)
    (grasped ?block - block)
    (smaller ?block - block ?location - location)
    (free-gripper)
  )

  (:action pick
    :parameters (?b - block ?l - location)
    :precondition (and (clear ?b) (on ?b ?l) (free-gripper))
    :effect (and (clear ?l) (not (on ?b ?l)) (grasped ?b) (not (free-gripper))))

  (:action place
    :parameters (?b - block ?l - location)
    :precondition (and (grasped ?b) (smaller ?b ?l) (clear ?l))
    :effect (and (not (grasped ?b)) (on ?b ?l) (not (clear ?l)) (free-gripper)))
)
