"""Domain knowledge: concept vocabulary, prerequisite graph, and corpus.

The paper applies CA-RAG to cognitive health education, using stroke risk and
prevention as the running example. This module encodes a small but coherent
stroke-education knowledge base:

* ``CONCEPTS``        -- the concept dimensions tracked by the Knowledge Vector
* ``PREREQUISITES``   -- a prerequisite graph over those concepts
* ``KEYWORDS``        -- lexical markers used by the rule-based judge to detect
                         whether a concept is mentioned/explained in a response
* ``EXPLANATIONS``    -- corpus passages for each concept at three difficulty
                         levels (beginner / intermediate / expert)

The corpus is intentionally self-contained so the framework can run and be
evaluated offline. In a production deployment the passages would be replaced by
documents from a real medical knowledge base (e.g. the PDF corpus in
``us-census-data`` for the vanilla demo, or authoritative clinical sources).
"""

from .config import BEGINNER, INTERMEDIATE, EXPERT

# --------------------------------------------------------------------------- #
# Concept vocabulary (ordered -- defines the KV dimensions)
# --------------------------------------------------------------------------- #
CONCEPTS = [
    "blood_circulation",
    "blood_vessels",
    "blood_pressure",
    "blood_clot",
    "ischemic_stroke",
    "hemorrhagic_stroke",
    "stroke_symptoms",
    "stroke_risk_factors",
    "hypertension",
    "stroke_prevention",
    "tia",
]

# --------------------------------------------------------------------------- #
# Prerequisite graph: concept -> list of concepts that should be understood
# before it. Used by Knowledge Gap Detection and by the Prerequisite Coverage
# Score (PCS).
# --------------------------------------------------------------------------- #
PREREQUISITES = {
    "blood_pressure": ["blood_circulation", "blood_vessels"],
    "blood_clot": ["blood_circulation"],
    "ischemic_stroke": ["blood_circulation", "blood_vessels", "blood_clot"],
    "hemorrhagic_stroke": ["blood_vessels", "blood_pressure"],
    "stroke_symptoms": ["ischemic_stroke", "hemorrhagic_stroke"],
    "hypertension": ["blood_pressure"],
    "stroke_risk_factors": ["hypertension", "blood_pressure", "blood_clot"],
    "stroke_prevention": ["stroke_risk_factors"],
    "tia": ["ischemic_stroke", "stroke_symptoms"],
    "blood_circulation": [],
    "blood_vessels": [],
}

# --------------------------------------------------------------------------- #
# Keyword markers used by the rule-based concept detector.
# --------------------------------------------------------------------------- #
KEYWORDS = {
    "blood_circulation": ["circulat", "blood flow", "pump", "oxygen", "heart"],
    "blood_vessels": ["arter", "vein", "blood vessel", "capillar"],
    "blood_pressure": ["blood pressure", "systolic", "diastolic", "pressure", "mmhg"],
    "blood_clot": ["clot", "coagulat", "blockage", "thromb"],
    "ischemic_stroke": ["ischemic", "block", "clot", "blood flow"],
    "hemorrhagic_stroke": ["hemorrhagic", "bleed", "ruptur", "hemorrhage"],
    "stroke_symptoms": ["fast", "face", "arm", "speech", "numb", "weakness", "symptom"],
    "stroke_risk_factors": ["risk factor", "smoking", "diabetes", "obes", "cholesterol", "inactive"],
    "hypertension": ["hypertension", "high blood pressure"],
    "stroke_prevention": ["prevent", "lifestyle", "diet", "exercise", "medication"],
    "tia": ["transient ischemic", "tia", "mini-stroke", "temporary"],
}

# --------------------------------------------------------------------------- #
# Corpus passages per concept per difficulty level.
# --------------------------------------------------------------------------- #
EXPLANATIONS = {
    "blood_circulation": {
        BEGINNER: (
            "In simple terms, blood circulation is how the heart pumps blood "
            "around the body. Think of the heart as a pump and the blood as "
            "water flowing through pipes. This flow carries oxygen to every "
            "part of the body and brings waste back to be removed."
        ),
        INTERMEDIATE: (
            "Blood circulation is the movement of blood through the heart and "
            "the network of blood vessels. The heart pumps oxygen-rich blood "
            "out through the arteries, and oxygen-poor blood returns through "
            "the veins. This circulation delivers oxygen and nutrients to "
            "tissues and removes carbon dioxide and waste products."
        ),
        EXPERT: (
            "Systemic circulation is driven by the cardiac cycle: the left "
            "ventricle ejects oxygenated blood into the aorta at a cardiac "
            "output of roughly 5 L/min, perfusing capillary beds where gas "
            "and solute exchange occurs, before deoxygenated blood returns via "
            "the venous system to the right atrium."
        ),
    },
    "blood_vessels": {
        BEGINNER: (
            "Blood vessels are the tubes that carry blood around the body. "
            "Imagine them as roads: arteries carry blood away from the heart, "
            "and veins carry it back. Small branches called capillaries reach "
            "every part of the body."
        ),
        INTERMEDIATE: (
            "Blood vessels form a closed circulatory network comprising "
            "arteries, veins, and capillaries. Arteries carry oxygenated blood "
            "away from the heart under high pressure, whereas veins return "
            "deoxygenated blood to the heart. Capillaries are the site of "
            "nutrient and gas exchange."
        ),
        EXPERT: (
            "The vascular tree consists of elastic arteries, muscular "
            "arterioles, exchange capillaries, and capacitance veins. "
            "Endothelial function and arterial compliance regulate resistance "
            "and pressure; endothelial dysfunction is an early step in "
            "atherosclerosis and hypertensive vascular disease."
        ),
    },
    "blood_pressure": {
        BEGINNER: (
            "Blood pressure is simply the force of blood pushing against the "
            "walls of your blood vessels. Imagine pressing your finger on a "
            "garden hose. When the pressure is too high for a long time, it "
            "can slowly damage the vessels."
        ),
        INTERMEDIATE: (
            "Blood pressure is the force exerted by circulating blood on the "
            "walls of blood vessels, measured as systolic pressure over "
            "diastolic pressure (for example 120/80 mmHg). Systolic pressure "
            "reflects contraction of the heart, while diastolic pressure "
            "reflects the resting phase between beats."
        ),
        EXPERT: (
            "Arterial blood pressure is the product of cardiac output and "
            "systemic vascular resistance, modulated by the renin-angiotensin-"
            "aldosterone system, sympathetic tone, and natriuretic peptides. "
            "Sustained hypertension drives endothelial injury, arterial "
            "stiffening, and target-organ damage."
        ),
    },
    "blood_clot": {
        BEGINNER: (
            "A blood clot is a lump that forms when blood thickens and sticks "
            "together. Normally this stops bleeding when you get a cut. But if "
            "a clot forms inside a blood vessel, it can block the flow of "
            "blood like a plug in a pipe."
        ),
        INTERMEDIATE: (
            "A blood clot (thrombus) forms through coagulation, in which "
            "platelets aggregate and fibrin strands cross-link to seal an "
            "injury. When a clot forms inside an intact vessel it can obstruct "
            "blood flow; if it breaks loose it becomes an embolus that travels "
            "until it lodges in a narrower vessel."
        ),
        EXPERT: (
            "Thrombogenesis involves the coagulation cascade, platelet "
            "activation via thromboxane A2 and ADP, and fibrin polymerisation. "
            "Pathological thrombosis arises from Virchow's triad of "
            "endothelial injury, stasis, and hypercoagulability, and is the "
            "proximal cause of arterial occlusion in ischemic stroke."
        ),
    },
    "ischemic_stroke": {
        BEGINNER: (
            "An ischemic stroke happens when a blood vessel to the brain gets "
            "blocked, usually by a clot. Because blood cannot reach part of "
            "the brain, that part starts to die. This is the most common type "
            "of stroke. Acting fast can save brain cells."
        ),
        INTERMEDIATE: (
            "Ischemic stroke occurs when an artery supplying the brain becomes "
            "blocked by a thrombus or embolus, depriving downstream tissue of "
            "oxygen. It accounts for roughly 87% of strokes. The ischemic "
            "penumbra (tissue at risk but salvageable) can be rescued with "
            "timely reperfusion therapy."
        ),
        EXPERT: (
            "Ischemic stroke results from focal cerebral hypoperfusion due to "
            "thrombotic or embolic arterial occlusion, causing a core of "
            "infarcted tissue surrounded by an ischemic penumbra. Outcomes "
            "depend on the time window to recanalisation via thrombolysis or "
            "mechanical thrombectomy, the collateral circulation, and penumbral "
            "salvage."
        ),
    },
    "hemorrhagic_stroke": {
        BEGINNER: (
            "A hemorrhagic stroke happens when a blood vessel in the brain "
            "bursts and leaks blood. The leaking blood damages nearby brain "
            "tissue. It is less common than a blocked-vessel stroke but is "
            "often more serious."
        ),
        INTERMEDIATE: (
            "Hemorrhagic stroke is caused by rupture of a cerebral blood "
            "vessel, leading to bleeding into or around the brain. It may "
            "result from high blood pressure, an aneurysm, or an arteriovenous "
            "malformation. The accumulating blood raises intracranial pressure "
            "and injures surrounding tissue."
        ),
        EXPERT: (
            "Intracerebral hemorrhage arises from rupture of small penetrating "
            "arteries, most often secondary to chronic hypertension or "
            "amyloid angiopathy, whereas subarachnoid hemorrhage is typically "
            "due to aneurysm rupture. Mass effect and elevated intracranial "
            "pressure drive secondary brain injury independent of the initial "
            "hematoma."
        ),
    },
    "stroke_symptoms": {
        BEGINNER: (
            "The warning signs of stroke can be remembered with the word FAST: "
            "Face drooping, Arm weakness, Speech difficulty, and Time to call "
            "for help. Other signs include sudden numbness, confusion, or "
            "trouble seeing. If you notice these, act fast."
        ),
        INTERMEDIATE: (
            "Stroke symptoms typically appear suddenly and include facial "
            "droop, arm weakness, slurred speech, and sometimes vision loss, "
            "dizziness, or severe headache. The FAST acronym summarises the "
            "core signs. Rapid recognition is essential because the benefit of "
            "reperfusion therapy is strongly time-dependent."
        ),
        EXPERT: (
            "Acute stroke presents with focal neurological deficits reflecting "
            "the affected vascular territory: motor or sensory loss, aphasia, "
            "dysarthria, visual field defects, and ataxia. Posterior-circulation "
            "events may present atypically with vertigo or diplopia, which "
            "contributes to diagnostic delay."
        ),
    },
    "stroke_risk_factors": {
        BEGINNER: (
            "Things that make a stroke more likely are called risk factors. "
            "The big ones you can change are smoking, high blood pressure, "
            "diabetes, being overweight, and not being active. Some risk "
            "factors, like age and family history, cannot be changed."
        ),
        INTERMEDIATE: (
            "Stroke risk factors are divided into modifiable and non-modifiable "
            "categories. Modifiable factors include hypertension, diabetes, "
            "dyslipidemia, obesity, physical inactivity, and smoking. "
            "Non-modifiable factors include age, sex, and family history. "
            "Controlling modifiable factors sharply lowers lifetime stroke risk."
        ),
        EXPERT: (
            "Modifiable vascular risk factors -- hypertension, diabetes "
            "mellitus, dyslipidemia, atrial fibrillation, obesity, physical "
            "inactivity, and tobacco use -- account for a large share of the "
            "population-attributable risk of stroke, and their control is the "
            "cornerstone of primary and secondary prevention."
        ),
    },
    "hypertension": {
        BEGINNER: (
            "Hypertension is the medical word for high blood pressure. It is "
            "often called the silent killer because you usually feel fine even "
            "though your blood pressure is too high. It is the most important "
            "risk factor for stroke."
        ),
        INTERMEDIATE: (
            "Hypertension, or persistently elevated blood pressure, is the "
            "leading modifiable risk factor for stroke. Over time, high "
            "pressure damages blood vessel walls and promotes atherosclerosis, "
            "increasing the risk of both ischemic and hemorrhagic stroke. "
            "Regular monitoring and treatment reduce this risk."
        ),
        EXPERT: (
            "Chronic hypertension induces endothelial dysfunction, arterial "
            "remodelling, and accelerated atherosclerosis, predisposing to "
            "both atherothrombotic ischemic stroke and small-vessel disease, "
            "and is the dominant modifiable risk factor for intracerebral "
            "hemorrhage."
        ),
    },
    "stroke_prevention": {
        BEGINNER: (
            "You can lower your chance of a stroke with simple daily choices: "
            "keep your blood pressure in check, eat a healthy diet, move your "
            "body regularly, stop smoking, and take medicines if your doctor "
            "prescribes them."
        ),
        INTERMEDIATE: (
            "Stroke prevention targets modifiable risk factors through blood "
            "pressure and cholesterol control, treatment of diabetes, smoking "
            "cessation, a healthy diet, regular physical activity, and weight "
            "management. Antiplatelet or anticoagulant medication is used in "
            "high-risk individuals."
        ),
        EXPERT: (
            "Primary prevention of stroke centres on blood-pressure reduction, "
            "lipid management with statins, glycaemic control, anticoagulation "
            "for atrial fibrillation, and lifestyle modification. Secondary "
            "prevention adds antiplatelet therapy and targeted management of "
            "the index stroke mechanism."
        ),
    },
    "tia": {
        BEGINNER: (
            "A TIA is sometimes called a mini-stroke or a warning stroke. The "
            "symptoms are like a stroke but last only a short time and go "
            "away. It is a serious warning sign that a bigger stroke could "
            "happen, so it still needs urgent medical attention."
        ),
        INTERMEDIATE: (
            "A transient ischemic attack (TIA) is a brief episode of stroke-like "
            "symptoms caused by a temporary reduction in blood flow to part of "
            "the brain, typically resolving within minutes to hours. Although "
            "symptoms resolve, a TIA is a strong warning sign of impending "
            "stroke and warrants urgent evaluation."
        ),
        EXPERT: (
            "Transient ischemic attack is defined by transient focal "
            "neurological symptoms without infarction on imaging. It confers a "
            "high short-term risk of subsequent ischemic stroke, and prompt "
            "evaluation (including ABCD2 stratification and vascular imaging) "
            "guides early secondary prevention."
        ),
    },
}

# Queries used to build the synthetic test set, mapped to their target concept.
QUERIES = {
    "What is a stroke?": "ischemic_stroke",
    "How does blood reach the brain?": "blood_circulation",
    "What is high blood pressure?": "blood_pressure",
    "Can high blood pressure cause a stroke?": "hypertension",
    "What are the warning signs of a stroke?": "stroke_symptoms",
    "What is a TIA or mini-stroke?": "tia",
    "How can I prevent a stroke?": "stroke_prevention",
    "What makes a stroke more likely?": "stroke_risk_factors",
    "What is the difference between a blocked and a bleeding stroke?": "hemorrhagic_stroke",
    "Why do blood clots cause strokes?": "blood_clot",
    "What are blood vessels?": "blood_vessels",
    "How does a clot block the brain's blood flow?": "ischemic_stroke",
}


def prerequisites_of(concept, closure=False):
    """Return the (direct or transitive) prerequisites of ``concept``.

    With ``closure=True`` the full transitive prerequisite set is returned,
    which is what the Prerequisite Coverage Score uses as the "task-required
    prerequisite" set.
    """
    if concept not in PREREQUISITES:
        return []
    if not closure:
        return list(PREREQUISITES[concept])

    seen = []
    stack = list(PREREQUISITES[concept])
    while stack:
        c = stack.pop()
        if c in seen:
            continue
        seen.append(c)
        stack.extend(PREREQUISITES.get(c, []))
    return seen


def explanation(concept, level):
    """Return the corpus passage for ``concept`` at ``level``."""
    return EXPLANATIONS.get(concept, {}).get(level, "")
