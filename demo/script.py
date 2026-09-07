"""The beat script: the film's single source of timing.

Each Beat carries the narration for one moment. Its synthesised length decides
how long the corresponding picture runs, so audio and video can never drift.
Beats are grouped by scene, in order.

`min_s` is a floor for beats whose picture needs room to breathe (a title
landing, a fault playing out) beyond the time the words take. `pad_s` is held
silence after the words, for a moment to register.

Run ``python -m demo.build --list`` after editing to see the new timings.
"""

from dataclasses import dataclass


@dataclass
class Beat:
    id:     str
    scene:  str
    text:   str
    min_s:  float = 0.0
    pad_s:  float = 0.0


BEATS = [
    # -- 0. Cold open ------------------------------------------------------
    Beat("b00", "s0",
         "Four drones. One hidden target. A thirty-two by thirty-two grid "
         "none of them can see.",
         min_s=6.0),
    Beat("b01", "s0",
         "And at any moment, any one of them can simply stop working.",
         min_s=5.5, pad_s=1.6),

    # -- 1. Setup ----------------------------------------------------------
    Beat("b10", "s1",
         "It starts here. Team size, how far each drone sees, how far it "
         "talks, how cluttered the world is, and how often drones fail.",
         min_s=9.0),
    Beat("b11", "s1",
         "Change any of them. The weights never change. One shared policy "
         "drives every drone, in every one of these worlds.",
         min_s=8.0, pad_s=0.6),

    # -- 2. Live search ----------------------------------------------------
    Beat("b20", "s2",
         "This is the team's combined knowledge. Everything dark has never "
         "been seen by anyone.",
         min_s=6.0),
    Beat("b21", "s2",
         "Nobody assigns territory. They spread out because re-sweeping a "
         "teammate's ground simply pays less.",
         min_s=6.5),
    Beat("b22", "s2",
         "Come within range, and two drones merge maps. One flash, and each "
         "knows everything the other knew.",
         min_s=6.5),
    Beat("b23", "s2",
         "But no drone ever sees the whole picture. These are the four "
         "private beliefs, and they disagree.",
         min_s=6.5),
    Beat("b24", "s2",
         "Coverage, reward, new ground, messages. Live, for the run you are "
         "watching.",
         min_s=5.5, pad_s=0.4),

    # -- 3. Fault ----------------------------------------------------------
    Beat("b30", "s3",
         "Now watch this one.",
         min_s=3.2),
    Beat("b31", "s3",
         "It is gone. It will not move, sense or transmit again, and its map "
         "is frozen at the moment it died.",
         min_s=7.0),
    Beat("b32", "s3",
         "There is no oracle. The others do not simply know. The wreck emits "
         "a beacon, and the news spreads one teammate at a time.",
         min_s=8.5),
    Beat("b33", "s3",
         "The team never stops. And the wreck still burns a turn every round, "
         "so a loss costs time as well as coverage.",
         min_s=7.5, pad_s=0.5),

    # -- 4. Inside the network --------------------------------------------
    Beat("b40", "s4",
         "So what does a drone actually decide with? Take this one, and open "
         "it up.",
         min_s=5.5),
    Beat("b41", "s4",
         "Its entire memory is six planes. Where it has been. The walls it "
         "found. How often it revisited each cell. The target, if it ever saw "
         "it. Its own position. And the crash sites it knows about.",
         min_s=13.5),
    Beat("b42", "s4",
         "Six thousand numbers. Not the world, but what this one drone "
         "believes about it.",
         min_s=6.0),
    Beat("b43", "s4",
         "Its position is never handed over as coordinates. It is a single "
         "one, in a plane of zeros. The network finds it, and crops a "
         "thirteen by thirteen window around it.",
         min_s=11.0),
    Beat("b44", "s4",
         "Thirteen: exactly wide enough for the largest vision radius it ever "
         "trained on.",
         min_s=6.0),
    Beat("b45", "s4",
         "Then five numbers: how far it sees, how far it talks, the team "
         "size, which drone it is, and how many teammates it believes are "
         "still alive.",
         min_s=11.0),
    Beat("b46", "s4",
         "That last one is a belief, not a fact. It can be wrong, and the "
         "policy acts on it anyway.",
         min_s=6.5),
    Beat("b47", "s4",
         "The wide map goes through a convolutional trunk and self attention, "
         "the patch through its own. The context steers every stage.",
         min_s=9.0),
    Beat("b48", "s4",
         "The head splits in two: how good this position is at all, and how "
         "much better each move is than the rest.",
         min_s=8.0),
    Beat("b49", "s4",
         "The largest is the move. One network, shared by every drone, "
         "deciding hundreds of times a mission.",
         min_s=8.0, pad_s=0.7),

    # -- 5. Generalization -------------------------------------------------
    Beat("b50", "s5",
         "One drone or six. Open ground or clutter. No failures, or failures "
         "three times likelier than anything it trained on.",
         min_s=9.0),
    Beat("b51", "s5",
         "Same weights. Every time.",
         min_s=4.0, pad_s=0.5),

    # -- 6. Outro ----------------------------------------------------------
    Beat("b60", "s6",
         "Teams of six it never trained on succeed ninety-nine times in a "
         "hundred. Across the whole trained failure range, success drops under "
         "seven points.",
         min_s=9.5),
    Beat("b61", "s6",
         "At triple the worst failure rate it ever saw, it still completes "
         "nearly three missions in four. It degrades. It does not collapse.",
         min_s=9.5, pad_s=1.6),
]


def by_scene():
    """Ordered {scene_id: [Beat, ...]}."""
    out = {}
    for b in BEATS:
        out.setdefault(b.scene, []).append(b)
    return out


SCENE_ORDER = ["s0", "s1", "s2", "s3", "s4", "s5", "s6"]

SCENE_TITLES = {
    "s0": (0, "Cold open"),
    "s1": (1, "Setup"),
    "s2": (2, "Cooperative search"),
    "s3": (3, "Failure"),
    "s4": (4, "Inside the policy"),
    "s5": (5, "Generalization"),
    "s6": (6, "Results"),
}
