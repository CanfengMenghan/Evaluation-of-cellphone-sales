def predict_proba(self, X): 
    try:
      rows=(X.shape[0])
    except:
      rows=len(X)
    X1 = self.build_matrix(X)
    if  self.k_models!=None and len(self.k_models)<2:
        predictions = self.bst.predict(X1)
    else :
        dtest = xgb.DMatrix(X)
        predictions= None
        for gbdt in self.k_models:
            predsnew = gbdt.predict(dtest, ntree_limit=(gbdt.best_iteration+1)*self.num_parallel_tree)  
            if predictions==None:
                predictions=predsnew
            else:
                for g in range (0, predsnew.shape[0]):
                    predictions[g]+=predsnew[g]
        for g in range (0, len(predictions)):
            predictions[g]/=float(len(self.k_models))               
        predictions=np.array(predictions)
    if self.objective == 'multi:softprob': return predictions.reshape( rows, self.num_class)
    return np.vstack([1 - predictions, predictions]).T 
