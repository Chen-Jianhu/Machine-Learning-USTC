function svm_iris()
  
% Load training features and labels
[ train_y , train_x ] = libsvmread ( 'iris_data_train.txt' );

gamma = 100;

% Libsvm options
% -s 0 : classification
% -t 2 : RBF kernel
% -g : gamma in the RBF kernel
model = svmtrain(train_y, train_x, sprintf('-s 0 -t 2 -g %g', gamma));

% Load testing features and labels
[ test_y , test_x ] = libsvmread ( 'iris_data_test.txt' );

% Display training accuracy
[predicted_label, accuracy, decision_values] = svmpredict(test_y, test_x, model);

% Plot training data and decision boundary
% plotboundary(y, x, model);
% title(sprintf('\\gamma = %g', gamma), 'FontSize', 14);
% =======================================================================

% Load training features and labels
[ train_y , train_x ] = libsvmread ( 'iris_data_train.txt' );

% Train the model and get the primal variables w, b from the model
% Libsvm options
% -t 0 : linear kernel
% Leave other options as their defaults
% model = svmtrain(train_y, train_x, '-t 0');
% w = model.SVs' * model.sv_coef;
% b = -model.rho;
% if (model.Label(1) == -1)
% w = -w; b = -b;
% end
model = svmtrain ( train_y , train_x , sprintf ( '-s 0 -t 0' ));

% Load testing features and labels
[ test_y , test_x ] = libsvmread ( 'iris_data_test.txt' );
[ predicted_label , accuracy , decision_values ] = svmpredict ( test_y , test_x , model );

% After running svmpredict, the accuracy should be printed to the matlab
% console


