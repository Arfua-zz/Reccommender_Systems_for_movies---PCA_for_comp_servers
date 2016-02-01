%%Anomaly Detection and Collaborative Filtering

%% =============== Part 1: Loading movie ratings dataset ================  
fprintf('Loading movie ratings dataset.\n\n');

load ('movies.mat');

%  From the matrix, we can compute statistics like average rating.
fprintf('Average rating for movie 1 (Toy Story): %f / 5\n\n', ...
        mean(Y(1, R(1, :))));

%  We can "visualize" the ratings matrix by plotting it with imagesc
imagesc(Y);
ylabel('Movies');
xlabel('Users');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ============ Part 2: Collaborative Filtering Cost Function ===========
load ('movieParams.mat');

%  Reduce the data set size so that this runs faster

num_users = 4; num_movies = 5; num_features = 3;
X = X(1:num_movies, 1:num_features);
Theta = Theta(1:num_users, 1:num_features);
Y = Y(1:num_movies, 1:num_users);
R = R(1:num_movies, 1:num_users);

%  Evaluate cost function
J = cofiCostFunc([X(:) ; Theta(:)], Y, R, num_users, num_movies, ...
               num_features, 0);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% ============== Part 3: Collaborative Filtering Gradient ==============
fprintf('\nChecking Gradients (without regularization) ... \n');

%  Check gradients by running checkNNGradients
checkCostFunction;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% ========= Part 4: Collaborative Filtering Cost Regularization ========
%  Evaluate cost function
J = cofiCostFunc([X(:) ; Theta(:)], Y, R, num_users, num_movies, ...
               num_features, 1.5);         

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% ======= Part 5: Collaborative Filtering Gradient Regularization ====== 
fprintf('\nChecking Gradients (with regularization) ... \n');

%  Check gradients by running checkNNGradients
checkCostFunction(1.5);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% ============== Part 6: Entering ratings for a new user ===============
movieList = loadMovieList();

%  Initialize my ratings
my_ratings = zeros(1682, 1);

% Check the file movie_idx.txt for id of each movie in dataset
% For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
my_ratings(1) = 4;

my_ratings(98) = 5;

% a few movies liked/disliked
my_ratings(2) = 3;
my_ratings(7) = 5;
my_ratings(12)= 5;
my_ratings(54) = 2;
my_ratings(64)= 5;
my_ratings(66)= 4;
my_ratings(69) = 5;
my_ratings(183) = 3;
my_ratings(226) = 5;
my_ratings(355)= 1;

fprintf('\n\nNew user ratings:\n');
for i = 1:length(my_ratings)
    if my_ratings(i) > 0 
        fprintf('Rated %d for %s\n', my_ratings(i), ...
                 movieList{i});
    end
end

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% ================== Part 7: Learning Movie Ratings ====================
fprintf('\nTraining collaborative filtering...\n');

load('movies.mat');

%  Add our own ratings to the data matrix
Y = [my_ratings Y];
R = [(my_ratings ~= 0) R];

%  Normalize Ratings
[Ynorm, Ymean] = normalizeRatings(Y, R);

%  Useful Values
num_users = size(Y, 2);
num_movies = size(Y, 1);
num_features = 10;

% Set Initial Parameters (Theta, X)
X = randn(num_movies, num_features);
Theta = randn(num_users, num_features);

initial_parameters = [X(:); Theta(:)];

% Set options for fmincg
options = optimset('GradObj', 'on', 'MaxIter', 100);

% Set Regularization
lambda = 10;
theta = fmincg (@(t)(cofiCostFunc(t, Y, R, num_users, num_movies, ...
                                num_features, lambda)), ...
                initial_parameters, options);

% Unfold the returned theta back into U and W
X = reshape(theta(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(theta(num_movies*num_features+1:end), ...
                num_users, num_features);

fprintf('Recommender system learning completed.\n');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ================== Part 8: Recommendation for you ====================
%  We can now make recommendations by computing the predictions matrix.

p = X * Theta';
my_predictions = p(:,1) + Ymean;

movieList = loadMovieList();

[r, ix] = sort(my_predictions, 'descend');
fprintf('\nTop recommendations for you:\n');
for i=1:10
    j = ix(i);
    fprintf('Predicting rating %.1f for movie %s\n', my_predictions(j), ...
            movieList{j});
end

fprintf('\n\nOriginal ratings provided:\n');
for i = 1:length(my_ratings)
    if my_ratings(i) > 0 
        fprintf('Rated %d for %s\n', my_ratings(i), ...
                 movieList{i});
    end
end
