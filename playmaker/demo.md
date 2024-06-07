 # The demo structure
 
*1.-* Introduce the dataset
   - use the excels as simple way to show the different datasets
   - Simple encodings
     - Flattened
     - Centroids
     - others possible, LSTM autoencoders

*2.-* Describe the ML models to be tested

  *Problem statement*: Play classification, organized game
    * centroids, flatten, ls* 
    *For LGBM, RanfomForest and KNN*: test with

     - centroids, distances
     - centroids, distances, time
     - centroids, distances, time + offense metadata (see that this is not getting really better)
     - centroids, distances, time + velocities, acceleration

  
  *Problem statement*: Possession success scoring, possession result
 
     *For LGBM, RanfomForest and KNN*: test with

      - centroids, distances (see it gets no where)
      - centroids, distances + offense metadata (now it does much better!)

    *show how the use_sequence=True* has an important effect in the results

*2.-* Describe the DL models to be tested

     * test with raw dataset
