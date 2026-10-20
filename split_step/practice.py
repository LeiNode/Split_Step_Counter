"""
practice.py — turns the coach's focus points into drills for the next session.

Each tip the analyses can produce is matched (by a phrase from its wording)
to one drill below. The plan takes the drills for the coach's focus tips in
priority order, without repeats. Tips with no drill (camera advice, "player
out of view") are skipped: they're about filming, not tennis.

When you add a tip anywhere, add its drill here; tests/test_practice.py
checks that every tip the analyses emit is covered.
"""

import re

# (pattern matched against the tip, drill); first match wins, so a pattern
# must come before any looser one that its tips also contain ("Recover
# faster: ... get back to a ready position" is a recovery-speed tip).
_DRILLS = [
    (r"split step before|without a split step|no split steps detected", {
        "name": "Split on the feed",
        "how": "Have a partner feed from the other baseline. Make a small two-footed hop exactly as their "
               "racket meets the ball, then move to the ball. Say \"hop\" out loud to lock in the timing.",
        "dose": "3 sets of 10 balls"}),
    (r"bend your knees more as you land|feet a bit wider|split steps are very small", {
        "name": "Split-step jumps",
        "how": "Without a ball: small hop, land on the balls of your feet slightly wider than your "
               "shoulders with knees bent, hold for one second. Then add a first step left or right.",
        "dose": "3 sets of 10 landings"}),
    (r"recover faster", {
        "name": "Two-ball recovery",
        "how": "Feeder sends one ball wide, then a second to the other side as soon as you hit. Recover to "
               "the middle with side-steps between the two.",
        "dose": "4 sets of 6 pairs"}),
    (r"ready position", {
        "name": "Hit and reset",
        "how": "After every fed ball, get back to the centre mark with knees bent and both hands on the "
               "racket before the next feed. The feeder waits until you're set.",
        "dose": "3 sets of 8 balls"}),
    (r"feet moving", {
        "name": "Small-step adjustment",
        "how": "Put a cone about a metre off the line of each feed. Run around it and take at least three "
               "small steps to set your distance before hitting.",
        "dose": "3 sets of 10 balls"}),
    (r"fade late", {
        "name": "Tennis intervals",
        "how": "Shadow footwork (split, move, swing, recover) for 30 s, rest 30 s. Builds the fitness to "
               "keep your footwork sharp at the end of a session.",
        "dose": "8 rounds, twice a week"}),
    (r"one shot\. work on consistency", {
        "name": "Rally to 20",
        "how": "Cooperative rally at 70% pace, aiming deep down the middle. Count out loud and restart at "
               "zero after a miss.",
        "dose": "Until you reach 20 in a row (cap at 15 minutes)"}),
    (r"changes a lot from shot to shot", {
        "name": "Groove one swing",
        "how": "Slow, full swings at 60% pace to the same target. Only speed up once 8 in a row land.",
        "dose": "3 sets of 12 balls"}),
    (r"making contact low", {
        "name": "Low-ball feeds",
        "how": "Feeder sends low, short balls. Bend from the knees (not the waist) so your eyes come down "
               "toward the ball; touch the court with your free hand on a few.",
        "dose": "3 sets of 8 balls"}),
    (r"above shoulder height", {
        "name": "Take it earlier",
        "how": "Feeder sends high-bouncing balls. Either step in and hit on the rise or back up and let it "
               "drop to chest height; never hit above your shoulders.",
        "dose": "3 sets of 8 balls"}),
    (r"turn your shoulders more", {
        "name": "Point and turn",
        "how": "Shadow swings: turn until your front shoulder points at the ball and your free hand points "
               "at it too, then swing. Repeat with fed balls.",
        "dose": "20 shadow swings, then 2 sets of 10 balls"}),
    (r"prepare earlier", {
        "name": "Turn on the bounce",
        "how": "Feeder calls \"forehand\" or \"backhand\" as they feed. Have your shoulders fully turned "
               "before the ball bounces on your side.",
        "dose": "3 sets of 10 balls"}),
    (r"head still", {
        "name": "Watch the strings",
        "how": "Keep your eyes on the contact point for a beat after every hit, as if you could still see "
               "the ball on your strings. Start with shadow swings, then fed balls.",
        "dose": "20 shadow swings, then 2 sets of 10 balls"}),
    (r"follow-through higher", {
        "name": "Catch the racket",
        "how": "Finish every swing over your opposite shoulder and catch the racket throat with your free "
               "hand before looking up.",
        "dose": "3 sets of 10 balls"}),
    (r"load your legs", {
        "name": "Sit and drive",
        "how": "Bend your knees as you set up and push up and through the ball. Exaggerate it at first: "
               "you should feel your thighs work.",
        "dose": "3 sets of 10 balls"}),
    (r"fully extend your hitting arm", {
        "name": "Reach for the sky",
        "how": "Serve with a high target in mind: make contact at full stretch with a straight arm, as if "
               "reaching up to a shelf above your head.",
        "dose": "3 sets of 8 serves"}),
    (r"trophy position", {
        "name": "Trophy pause",
        "how": "Toss, stop in the trophy position for a second with your knees bent, then drive up and "
               "serve. The pause makes the leg drive easy to feel.",
        "dose": "3 sets of 8 serves"}),
    (r"toss a little higher|toss release point", {
        "name": "Toss-only practice",
        "how": "Place a racket on the court in front of your front foot. Toss without hitting and let the "
               "ball land on the strings, releasing from the same spot at eye height each time.",
        "dose": "30 tosses"}),
]
_COMPILED = [(re.compile(p, re.IGNORECASE), d) for p, d in _DRILLS]


def drill_for(tip: str) -> dict | None:
    """The drill for one tip, or None (filming advice, praise)."""
    for pattern, drill in _COMPILED:
        if pattern.search(tip):
            return drill
    return None


def practice_plan(focus: list[str]) -> list[dict]:
    """Drills for the focus tips in priority order, each drill once:
    [{name, how, dose, for}], where "for" is the tip it addresses."""
    plan, seen = [], set()
    for tip in focus:
        drill = drill_for(tip)
        if drill and drill["name"] not in seen:
            seen.add(drill["name"])
            plan.append({**drill, "for": tip})
    return plan
