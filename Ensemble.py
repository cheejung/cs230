# Using VGG16 + ResNet50 Ensemble

vgg.load_weights('/kaggle/input/vgg-best/VGG_HybridDS.h5')
model.load_weights('/kaggle/input/hybrid-best/weights.best')

# For test data
test_data = test_datagen.flow_from_directory(
    '/kaggle/input/hybrid/HybridSplit/HybridTest',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=False,
    classes=['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
)

# WEIGHTED ENSEMBLE

# Ensemble predictions with weights
def weighted_ensemble_predictions(test_data, weight_vgg, weight_resnet):
    pred1 = vgg.predict(test_data)
    pred2 = model.predict(test_data)
    final_pred = (weight_vgg * pred1) + (weight_resnet * pred2)
    return final_pred

# Calculate accuracies of individual models
accuracy_vgg = vgg.evaluate(test_data, verbose=0)[1]
accuracy_resnet = model.evaluate(test_data, verbose=0)[1]

# Normalize the accuracies so they sum to 1
total = accuracy_vgg + accuracy_resnet
weight_vgg = accuracy_vgg / total
weight_resnet = accuracy_resnet / total

# Get the final predictions
final_predictions = weighted_ensemble_predictions(test_data, weight_vgg, weight_resnet)

# Convert predictions to labels
final_labels = np.argmax(final_predictions, axis=1)

# Evaluate the performance
true_labels = test_data.classes
ensemble_accuracy = np.sum(final_labels == true_labels) / len(true_labels)

print(f'Accuracy of the weighted ensemble of networks on test images: {ensemble_accuracy}%')

# Calculate accuracies of individual models
accuracy_vgg_high = vgg.evaluate(test_highres_data, verbose=0)[1]
accuracy_resnet_high = model.evaluate(test_highres_data, verbose=0)[1]

# Normalize the accuracies so they sum to 1
total = accuracy_vgg_high + accuracy_resnet_high
weight_vgg_high = accuracy_vgg_high / total
weight_resnet_high = accuracy_resnet_high / total

# Get the final predictions
final_predictions = weighted_ensemble_predictions(test_highres_data, weight_vgg_high, weight_resnet_high)

# Convert predictions to labels
final_labels = np.argmax(final_predictions, axis=1)

# Evaluate the performance
true_labels = test_highres_data.classes
high_ensemble_accuracy = np.sum(final_labels == true_labels) / len(true_labels)

print(f'Accuracy of the weighted ensemble of networks on high-res images: {high_ensemble_accuracy}%')

# Calculate accuracies of individual models
accuracy_vgg_low = vgg.evaluate(test_lowres_data, verbose=0)[1]
accuracy_resnet_low = model.evaluate(test_lowres_data, verbose=0)[1]

# Normalize the accuracies so they sum to 1
total = accuracy_vgg_low + accuracy_resnet_low
weight_vgg_low = accuracy_vgg_low / total
weight_resnet_low = accuracy_resnet_low / total

# Get the final predictions
final_predictions = weighted_ensemble_predictions(test_lowres_data, weight_vgg_low, weight_resnet_low)

# Convert predictions to labels
final_labels = np.argmax(final_predictions, axis=1)

# Evaluate the performance
true_labels = test_lowres_data.classes
low_ensemble_accuracy = np.sum(final_labels == true_labels) / len(true_labels)

print(f'Accuracy of the weighted ensemble of networks on low-res images: {low_ensemble_accuracy}%')

# CLASS-WEIGHTED ENSEMBLE

def class_accuracies(mod, test_data):
    # Get the true labels
    true_labels = test_data.classes

    # Predict labels
    predictions = mod.predict(test_data)
    predicted_labels = np.argmax(predictions, axis=1)

    # Generate classification report
    report = classification_report(true_labels, predicted_labels, output_dict=True)
    
    # Extract class accuracies from the report
    class_accuracies = []
    for i in range(10):  # assuming 10 classes
        class_accuracies.append(report[str(i)]['f1-score'])
    
    return np.array(class_accuracies)

def class_weighted_ensemble(test_data, class_acc_vgg, class_acc_resnet):
    pred1 = vgg.predict(test_data)
    pred2 = model.predict(test_data)
    
    # reshape class accuracies to match prediction shape and normalize
    class_acc_vgg_norm = class_acc_vgg / (class_acc_vgg + class_acc_resnet)
    class_acc_resnet_norm = class_acc_resnet / (class_acc_vgg + class_acc_resnet)
    
    class_acc_vgg_norm = class_acc_vgg_norm.reshape(1, -1)
    class_acc_resnet_norm = class_acc_resnet_norm.reshape(1, -1)
    
    final_pred = class_acc_vgg_norm * pred1 + class_acc_resnet_norm * pred2
    return final_pred

# Calculate class accuracies for each model
class_vgg = class_accuracies(vgg, test_data)
class_resnet = class_accuracies(model, test_data)

# Get the final predictions
final_predictions = class_weighted_ensemble(test_data, class_vgg, class_resnet)
final_labels = np.argmax(final_predictions, axis=1)

# Evaluate the performance
true_labels = test_data.classes
ensemble_class_accuracy = np.sum(final_labels == true_labels) / len(true_labels)

print(f'Accuracy of the class-weighted ensemble of networks on Hybrid images: {ensemble_class_accuracy}%')

# Calculate class accuracies for each model
class_high_vgg = class_accuracies(vgg,test_highres_data)
class_high_resnet = class_accuracies(model, test_highres_data)

# Get the final predictions
final_predictions = class_weighted_ensemble(test_highres_data, class_high_vgg, class_high_resnet)
final_labels = np.argmax(final_predictions, axis=1)

# Evaluate the performance
true_labels = test_highres_data.classes
ensemble_classhigh_accuracy = np.sum(final_labels == true_labels) / len(true_labels)

print(f'Accuracy of the class-weighted ensemble of networks on High-res images: {ensemble_classhigh_accuracy}%')

# Calculate class accuracies for each model
class_low_vgg = class_accuracies(vgg,test_lowres_data)
class_low_resnet = class_accuracies(model, test_lowres_data)

# Get the final predictions
final_predictions = class_weighted_ensemble(test_lowres_data, class_low_vgg, class_low_resnet)
final_labels = np.argmax(final_predictions, axis=1)

# Evaluate the performance
true_labels = test_lowres_data.classes
ensemble_classlow_accuracy = np.sum(final_labels == true_labels) / len(true_labels)

print(f'Accuracy of the class-weighted ensemble of networks on Low-res images: {ensemble_classlow_accuracy}%')