"""Central configuration for the CA-RAG framework.

All tunable constants that appear in the paper are collected here so that the
cognitive model, gap detection, and adaptive generation share a single source
of truth.
"""

# --------------------------------------------------------------------------- #
# Cognitive levels
# --------------------------------------------------------------------------- #
BEGINNER = 0
INTERMEDIATE = 1
EXPERT = 2

LEVEL_NAMES = {
    BEGINNER: "Beginner",
    INTERMEDIATE: "Intermediate",
    EXPERT: "Expert",
}

# The maximum possible difference between a user level and a response
# difficulty level. Used to normalise the Cognitive Alignment Score (CAS) so
# that it lies in [0, 1].
MAX_LEVEL_DIFF = 2

# --------------------------------------------------------------------------- #
# Cognitive Modeling (CM)
# --------------------------------------------------------------------------- #
# Weighting coefficients for the simplified KV update rule
#   k_i' = k_i + alpha * P_i - beta * Q_i - gamma * D_i
# where P_i is a successful-understanding signal, Q_i a clarification request,
# and D_i an excessive-dwell signal.
ALPHA = 0.10
BETA = 0.15
GAMMA = 0.10

# A concept is considered "missing" when its mastery score falls below this
# threshold. Used by Knowledge Gap Detection to decide which prerequisites
# require explanation.
MISSING_THRESHOLD = 0.5

# --------------------------------------------------------------------------- #
# Knowledge Gap Detection (KGD)
# --------------------------------------------------------------------------- #
# The Knowledge Gap Vector is derived as KGV = 1 - KV.

# --------------------------------------------------------------------------- #
# Adaptive Generation (AG)
# --------------------------------------------------------------------------- #
# Mastery threshold (tau) above which the user is considered "ready" for the
# advanced-explanation stage.
ADVANCED_READINESS_TAU = 0.7

# --------------------------------------------------------------------------- #
# Interaction signals
# --------------------------------------------------------------------------- #
# Dwell time (seconds) above which a user is flagged for "excessive dwell
# time" on a concept. In a real deployment this would be estimated from pilot
# interaction data; here we use a fixed default.
DWELL_THRESHOLD = 20.0

# --------------------------------------------------------------------------- #
# Dataset construction
# --------------------------------------------------------------------------- #
# Train / validation / test split ratio described in the paper.
SPLIT_RATIOS = (0.8, 0.1, 0.1)

# --------------------------------------------------------------------------- #
# Paths (relative to the repository root)
# --------------------------------------------------------------------------- #
# Original census PDF corpus used by the vanilla ObjectBox/LangChain demo.
CENSUS_DATA_DIR = "us-census-data"
# ObjectBox database directory used by the real vector-store backend.
OBJECTBOX_DIR = "objectbox"
