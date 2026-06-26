(define (problem stacking_3)
  (:domain stacking)
  (:objects
    cube1 cube2 cube3 - block
    table - table
  )
  (:init
    (clear cube1)
    (clear cube2)
    (clear cube3)
    (free-gripper)
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
