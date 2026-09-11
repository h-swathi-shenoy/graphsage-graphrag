"""Practical example corpus: flight-operations knowledge base of a
fictional airline, "SkyLink Airlines".

Deliberately designed so that useful answers require MULTI-HOP joins across
documents (flight -> shared aircraft -> shared gate/ramp scheduler ->
staffing change that broke it -> safety policy -> team -> incident runbook).
Plain vector RAG retrieves the one chunk that lexically matches the question
and misses the causal chain; GraphRAG + GraphSAGE walks the chain.
"""

DOCS = [
    ("flight-202", """Flight 202 is SkyLink Airlines' daily ORD-to-LAX
    route. It is flown by Aircraft N882AA and depends on ramp-scheduler for
    gate turnaround slots and on crew-scheduler for pilot and flight
    attendant assignment. Flight 202 is owned by Team Flight Operations.
    Its on-time performance is tracked against the Turnaround-Time SLA."""),

    ("aircraft-n882aa", """Aircraft N882AA is a shared Boeing 737 based at
    ORD used to fly Flight 202 and two other routes, Flight 118 to DFW and
    Flight 305 to SEA, on rotation the same day. It is maintained by Team
    Ground Operations and requires a minimum 45-minute standard turnaround
    between flights. N882AA occupies a slot in ramp-scheduler every time it
    turns around at the gate."""),

    ("sys-crewscheduler", """crew-scheduler is SkyLink's crew scheduling
    system. It assigns pilots and flight attendants to flights while
    enforcing FAA duty-time rules, and is owned by Team Crew Scheduling. It
    reads the roster queue from crew-db and can pull a backup crew member
    from a reserve pool when a crew member times out mid-duty."""),

    ("proc-extmarshal", """The extended marshalling procedure is a
    ground-safety check Team Ground Operations can run on an aircraft
    between turnarounds. Normally it is reserved for aircraft flagged after
    a mechanical hold and takes 90 minutes, well beyond the standard
    45-minute turnaround. It requires the aircraft to remain at the gate at
    ORD until ops sign-off."""),

    ("ramp-sched", """ramp-scheduler is the gate and ground-handling
    scheduling system for SkyLink's ORD hub, operated by Team Airport
    Operations. It has a hard limit of 8 concurrent gate turnaround slots
    per hour. Both routine turnarounds and extended marshalling holds for
    aircraft like N882AA occupy a slot in ramp-scheduler until released."""),

    ("inc-2209", """Flight delay incident INC-2209 occurred on 2025-03-08.
    Flight 202 departed 3 hours late. The immediate symptom was Aircraft
    N882AA held at the gate pending ground-crew sign-off at ORD, delaying
    three other flights queued behind it. Severity was Major and the
    estimated cost of rebooking and crew overtime was $180,000."""),

    ("inc-2209-rca", """Root cause analysis for INC-2209: Team Ground
    Operations had rolled out Staffing Policy SP-118, which mandated the
    90-minute extended marshalling check on every aircraft's first
    turnaround of the day instead of only aircraft flagged after a
    mechanical hold. N882AA's extended marshalling held its slot in
    ramp-scheduler well past the 8-slot hourly capacity, and the resulting
    backlog cascaded into three downstream flights. Flight 202 was the
    first flight caught by the change, which is why INC-2209 was raised."""),

    ("inc-1187", """Flight delay incident INC-1187 occurred on 2025-01-19.
    crew-scheduler assigned a pilot to a flight, but the pilot hit an FAA
    duty-time limit mid-duty because of an earlier upstream delay. Team
    Crew Scheduling pulled a backup pilot from the reserve pool in crew-db
    and the flight departed 2 hours late. Severity was Minor. The fix was
    tracked in Staffing Policy SP-95, adding an automatic duty-time-buffer
    check."""),

    ("capa-turnaround", """Incident runbook "Gate turnaround backlog at
    ORD": page Team Airport Operations, check the ramp-scheduler
    utilization dashboard, and identify which aircraft are overstaying
    their turnaround slots. Coordinate with Team Ground Operations to
    expedite ops sign-off or reassign the held aircraft to an overflow gate
    slot. Do not cancel downstream flights first: reassigning the slot
    resolves the backlog faster and is the designated response for
    INC-2209-style events."""),

    ("capa-crewhold", """Incident runbook "Crew duty-time hold": pull a
    qualified backup crew member from the reserve pool in crew-db, notify
    Team Crew Scheduling, and confirm rest-compliance documentation before
    the outgoing crew member's next assignment. Owned by Team Crew
    Scheduling."""),

    ("team-groundops", """Team Ground Operations is a fourteen-person team
    led by Marcus Webb. They own aircraft turnaround upkeep, including
    N882AA's marshalling cycles, and issue staffing-policy directives such
    as SP-118. Their on-call rotation is ground-ops-oncall in PagerDuty.
    Marcus Webb also chairs SkyLink's Operations Control Board."""),

    ("team-airportops", """Team Airport Operations, led by Dana Okoro,
    operates shared ORD hub infrastructure, including ramp-scheduler and
    gate turnaround logistics. They own the incident runbook "Gate
    turnaround backlog at ORD" and enforce the 8-slot hourly capacity
    policy."""),

    ("team-flightops", """Team Flight Operations, led by Priya Nair, owns
    Flight 202 and SkyLink's published on-time schedules. They are the
    primary stakeholder for the Turnaround-Time SLA and raised INC-2209
    after the March 8th delay."""),

    ("policy-faa7", """SkyLink safety policy PL-FAA-7 states that any
    marshalling procedure expected to exceed the standard 45-minute
    turnaround by more than 15 minutes must be scheduled during planned
    downtime, not during live flight operations, unless an exception is
    filed with the Operations Control Board. Staffing Policy SP-118 was
    rolled out without filing such an exception and is non-compliant with
    PL-FAA-7."""),

    ("sp-118", """Staffing Policy SP-118, titled "Proactive extended
    marshalling for ramp-safety inspection", changed the extended
    marshalling trigger from "aircraft flagged after a mechanical hold" to
    "every aircraft's first turnaround of the day". It was reviewed by one
    supervisor on Team Ground Operations and did not go through SkyLink's
    Operations Control Board before rollout."""),

    ("sla-turnaround", """The Turnaround-Time SLA defines SkyLink's target
    that 95% of flights depart within 15 minutes of scheduled time,
    measured across the ORD hub. Missing the weekly target automatically
    triggers an executive review and pages both ground-ops-oncall and Team
    Flight Operations."""),
]


def get_docs():
    return [(doc_id, " ".join(text.split())) for doc_id, text in DOCS]
