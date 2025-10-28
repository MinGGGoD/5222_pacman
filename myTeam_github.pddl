; Pacman Capture the Flag domain for classical planning

(define (domain pacman_domain)
    (:requirements :strips :typing)
    ;(:types pacman ghost location - object)
    (:types pacman ghost location)

    
    ;; Predicates
    (:predicates
        (at ?a - pacman ?l - location) ; our pacman is at location
        (ghostAt ?l - location) ; a ghost occupies this location
        (foodAt ?l - location) ; food pellet exists at this location
        (capsuleAt ?l - location) ; power capsule exists at this location
        (adjacent ?l1 - location ?l2 - location) ; locations are adjacent and traversable
        (carrying) ; our pacman is carrying at least one food pellet
        (capsulesEaten) ; a capsule has been eaten
        (wantsDie) ; pacman commits suicide by entering a ghost location
    )
    
    ;; Move to an adjacent location that is not occupied by a ghost
    (:action move
        :parameters (?p - pacman ?from - location ?to - location)
        :precondition (and 
            (at ?p ?from)
            (adjacent ?from ?to)
            (not (ghostAt ?to))
        )
        :effect (and
            (not (at ?p ?from))
            (at ?p ?to)
        )
    )

    ;; Move to an adjacent location regardless of ghosts (used for when pacman is scared or fleeing)
    (:action move-no-restriction
        :parameters (?p - pacman ?from - location ?to - location)
        :precondition (and 
            (at ?p ?from)
            (adjacent ?from ?to)
        )
        :effect (and
            (not (at ?p ?from))
            (at ?p ?to)
        )
    )

    ;; Eat a food pellet on the current location
    (:action eat-food
        :parameters (?p - pacman ?l - location)
        :precondition (and 
            (at ?p ?l)
            (foodAt ?l)
        )
        :effect (and
            (not (foodAt ?l))
            (carrying)
        )
    )

    ;; Eat a power capsule on the current location
    (:action eat-capsule
        :parameters (?p - pacman ?l - location)
        :precondition (and
            (at ?p ?l)
            (capsuleAt ?l)
        )
        :effect (and
            (not (capsuleAt ?l))
            (capsulesEaten)
        )
    )

    ;; Commit suicide by intentionally moving into a ghost
    (:action get-eaten
        :parameters (?p - pacman ?l - location)
        :precondition (and
            (at ?p ?l)
            (ghostAt ?l)
            (wantsDie)
        )
        :effect (and
            (not (carrying))
        )
    )

    ;; Defensive action for a ghost catching a pacman at a location
    (:action kill-invader
        :parameters (?g - ghost ?p - pacman ?l - location)
        :precondition (and
            (ghostAt ?l)
            (at ?p ?l)
        )
        :effect (and
            (not (at ?p ?l))
        )
    )
)