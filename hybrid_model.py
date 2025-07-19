class HybridRegressor:
    def __init__(self, model1, model2, weight1=0.6, weight2=0.4):
        self.model1 = model1
        self.model2 = model2
        self.weight1 = weight1
        self.weight2 = weight2

    def predict(self, X):
        preds1 = self.model1.predict(X)
        preds2 = self.model2.predict(X)
        return self.weight1 * preds1 + self.weight2 * preds2
