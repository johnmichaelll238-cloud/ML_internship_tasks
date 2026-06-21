from sklearn.tree import DecisionTreeClassifier

try:
    from .config import CRITERION, RANDOM_STATE
except ImportError:
    from config import CRITERION, RANDOM_STATE


def train_baseline(X_train, y_train):
    """
    TRAIN UNPRUNED BASELINE DECISION TREE
    RETURN TRAINED BASELINE DECISION TREE MODEL
    """
    model = DecisionTreeClassifier(
        criterion=CRITERION,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    return model


def generate_pruning_path(model, X_train, y_train):
    """
    GENERATE CANDIDATE CCP ALPHAS FOR PRUNING
    RETURNS NUMPY ARRAY, np.ndarray: Canidate ccp alpha values
    """
    path = model.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas

    return ccp_alphas


def train_candidate_models(X_train, y_train, ccp_alphas) -> list:
    candidate_models = []
    for alpha in ccp_alphas:
        tree = DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            criterion=CRITERION,
            ccp_alpha=alpha,
        )
        tree.fit(X_train, y_train)

        candidate_models.append(tree)

    return candidate_models


def train_final_model(X_train, y_train, best_alpha):
    model = DecisionTreeClassifier(
        criterion=CRITERION,
        random_state=RANDOM_STATE,
        ccp_alpha=best_alpha,
    )
    model.fit(X_train, y_train)
    return model
