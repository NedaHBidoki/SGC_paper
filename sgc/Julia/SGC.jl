
using Flux
using Flux: crossentropy, normalise, onecold, onehotbatch
using Statistics: mean
using StatsBase

pplN = 3
varN = 2
ADJ = [0 1 1; 1 0 0; 1 0 0]
#make an S matrix that is the norm of ADJ!!!!



EPOCHS = 8
N = 150
data = rand(pplN*varN,N)
normed_data = normalise(data, dims=2)
println(size(normed_data))
states = ["Happy","Sad"]
H_Threshold = -1#percentile(vec(mean(normed_data,dims=[1])),0.01)

function f_H_S(xvec)
if( sum(xvec) < H_Threshold ) 
    return states[2]
else
    return states[1]
end
end

labelsGCNN = vec(mapslices(x-> f_H_S(x) ,normed_data,dims=[1]))
println(size(labelsGCNN))
println(labels)
klassesGCNN = sort(unique(states))
train_indices = [1:3:N; 2:3:N]
println(size(train_indices))
onehot_labelsGCNN = onehotbatch(labelsGCNN, klassesGCNN)
println(onehot_labelsGCNN)
X_train = normed_data[:, train_indices]
X_test = normed_data[:, 3:3:N]
y_train = onehot_labelsGCNN[:, train_indices]
y_test = onehot_labelsGCNN[:, 3:3:N]
println(size(X_train))
println(size(X_test))
println(size(y_train))
println(size(y_test))
######################


model = model2

theta = param(rand(length(states),1))
W1 = param(rand(length(states), pplN*varN))
function gcnn(xvec)   
    return W1 * xvec .+ theta
end
model2(x) = softmax(gcnn(x))

loss(x, y) = crossentropy(model(x), y)


####################
# Gradient descent optimiser with learning rate 0.5.
optimiser = Descent(0.5)
# Create iterator to train model over 110 epochs.
data_iterator = Iterators.repeated((X_train, y_train), EPOCHS)
println("Starting training.")
#Flux.train!(loss, params(model), data_iterator, optimiser)
Flux.train!(loss, params(W1, theta), data_iterator, optimiser)
# Evaluate trained model against test set.
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))
accuracy_score = accuracy(X_test, y_test)
println("\nAccuracy: $accuracy_score")
# Sanity check.
#@assert accuracy_score > 0.8
function confusion_matrix(X, y)
    y2 = onehotbatch(onecold(model(X)), 1:2)
    y * y2'
end
#To avoid confusion, here is the definition of a Confusion Matrix: https://en.wikipedia.org/wiki/Confusion_matrix
println("\nConfusion Matrix:\n")
display(confusion_matrix(X_test, y_test))



#model3 = Chain(Dense(6, 2),softmax)

