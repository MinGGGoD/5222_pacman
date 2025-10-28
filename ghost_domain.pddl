; Ghost-specific PDDL domain for Capture the Flag
; This domain models actions relevant to a defensive ghost agent

(define (domain ghost_domain)
    (:requirements :strips :typing)

    ;; Types: ghost, pacman and location.  Ghost agents pursue pacmen
    (:types ghost pacman location)

    ;; Predicates
    (:predicates
        ; Ghost is at a location
        (at ?g - ghost ?l - location)
        ; Enemy pacman is at a location (for tracking invaders)
        (pacmanAt ?l - location)
        ; Traversable adjacency relation
        (adjacent ?l1 - location ?l2 - location)
    )

    ;; Move ghost to an adjacent location
    (:action move
        :parameters (?g - ghost ?from - location ?to - location)
        :precondition (and
            (at ?g ?from)
            (adjacent ?from ?to)
        )
        :effect (and
            (not (at ?g ?from))
            (at ?g ?to)
        )
    )

    ;; Eliminate an invader pacman at the same location
    (:action kill-invader
        :parameters (?g - ghost ?p - pacman ?l - location)
        :precondition (and
            (at ?g ?l)
            (pacmanAt ?l)
        )
        :effect (and
            ; Remove invader from this location
            (not (pacmanAt ?l))
        )
    )
)