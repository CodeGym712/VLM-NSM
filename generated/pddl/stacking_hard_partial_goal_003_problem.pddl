(define (problem stacking_hard_partial_goal_003_problem)
  (:domain stacking)
  (:objects
    cube1 cube2 cube3 - block
    table - table
  )
  (:init
    (free-gripper)
    (on cube1 table)
    (clear cube1)
    (on cube2 table)
    (clear cube2)
    (on cube3 table)
    (clear cube3)
    (smaller cube1 cube2)
    (smaller cube1 cube3)
    (smaller cube1 table)
    (smaller cube2 cube3)
    (smaller cube2 table)
    (smaller cube3 table)
  )
  (:goal
    (and
      (on cube1 cube2)
      (on cube2 cube3)
      (on cube3 table)
    )
  )
)
