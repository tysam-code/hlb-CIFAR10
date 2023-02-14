import torchvision

model_names = []
failed_model_names = []
for model_name in torchvision.models.list_models():
    try:
        torchvision.models.get_model(model_name, weights=None, num_classes=10)
    except:
        failed_model_names.append(model_name)
    else:
        model_names.append(model_name)


print("failed_model_names:", len(failed_model_names), failed_model_names)
print("model_names:", len(model_names), model_names)
