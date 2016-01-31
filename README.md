# Reccommender-Systems-for-movies-PCA-for-comp-servers

The project was finished as a part of Machine Learning course by Standford University.
Implementation is in MATLAB.
Project consists of two separated parts: 
- Principal component analysis for computer servers (PrincipalComponentAnalysis.m)
An anomaly detection algorithm is built to detect anomalous behavior in server computers. The features measure the through-put (mb/s)
and latency (ms) of response of each server. While servers were operating, we collected m = 307 examples of how they were behaving,
and thus have an unlabeled dataset. The vast majority of these examples are \normal" (non-anomalous) examples of the servers
operating normally, but there might also be some examples of servers acting anomalously within this dataset.
- Reccommender system for rating movies (RecommenderSystemMovies.m)
The collaborative filtering learning algorithm was built and applied to a dataset of movie ratings. This dataset consists
of ratings on a scale of 1 to 5. The dataset has nu = 943 users, and nm = 1682 movies. 
