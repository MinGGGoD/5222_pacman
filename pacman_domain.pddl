; Pacman-specific PDDL domain for Capture the Flag
; This domain models only actions relevant to our Pacman agent

(define (domain pacman_domain)
    (:requirements :strips :typing)

    ;; Declare primitive types.  These types are treated as subtypes of
    ;; the built‑in 'object' type.  Explicit enumeration avoids piglet
    ;; parser issues with cascading "- object" syntax.
    (:types pacman ghost location)

    ;; Predicates
    (:predicates
        ; Our pacman is at a location
        (at ?p - pacman ?l - location)
        ; A ghost occupies a location (for collision checks)
        (ghostAt ?l - location)
        ; Food pellet exists at a location
        (foodAt ?l - location)
        ; Power capsule exists at a location
        (capsuleAt ?l - location)
        ; Traversable adjacency relation
        (adjacent ?l1 - location ?l2 - location)
        ; Pacman is carrying at least one food pellet
        (carrying)
        ; Pacman has eaten a capsule
        (capsulesEaten)
        ; Pacman intentionally wants to die (suicide move)
        (wantsDie)
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

    ;; Move to an adjacent location regardless of ghosts (used when fleeing)
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

    ;; Eat a food pellet at the current location
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

    ;; Eat a power capsule at the current location
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

    ;; Commit suicide by intentionally staying in a ghost location
    (:action get-eaten
        :parameters (?p - pacman ?l - location)
        :precondition (and
            (at ?p ?l)
            (ghostAt ?l)
            (wantsDie)
        )
        :effect (and
            ; After death we drop all carried food
            (not (carrying))
        )
    )
)