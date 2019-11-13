##### CATS AND DOGS 

original_dataset_dir <- "/home/liam_local/Documents/Machine_learning/Dogs_cats/train"
base_dir <- "/home/liam_local/Documents/Machine_learning/Dogs_cats"

train_dir <- file.path(base_dir, "train_data")
dir.create(train_dir)
validation_dir <- file.path(base_dir, "validation")
dir.create(validation_dir)
test_dir <- file.path(base_dir, "test_data")
dir.create(test_dir)

train_cats_dir <- file.path(train_dir, "cats")
dir.create(train_cats_dir)
train_dogs_dir <- file.path(train_dir, "dogs")
dir.create(train_dogs_dir)

validation_cats_dir <- file.path(validation_dir, "cats")
dir.create(validation_cats_dir)
validation_dogs_dir <- file.path(validation_dir, "dogs")
dir.create(validation_dogs_dir)

test_cats_dir <- file.path(test_dir, "cats")
dir.create(test_cats_dir)
test_dogs_dir <- file.path(test_dir, "dogs")
dir.create(test_dogs_dir)

fnames <- paste0("cat.", 1:1000, ".jpg") # to training file
file.copy(file.path(original_dataset_dir, fnames), file.path(train_cats_dir))

fnames <- paste0("cat.", 1001:1500, ".jpg") # to validation file
file.copy(file.path(original_dataset_dir, fnames), file.path(validation_cats_dir))

fnames <- paste0("cat.", 1501:2000, ".jpg")
file.copy(file.path(original_dataset_dir, fnames), file.path(test_cats_dir))

fnames <- paste0("dog.", 1:1000, ".jpg")
file.copy(file.path(original_dataset_dir, fnames), file.path(train_dogs_dir))

fnames <- paste0("dog.", 1001:1500, ".jpg")
file.copy(file.path(original_dataset_dir, fnames), file.path(validation_dogs_dir))

fnames <- paste0("dog.", 1501:2000, ".jpg")
file.copy(file.path(original_dataset_dir, fnames), file.path(test_dogs_dir))

cat("total training cat images:", length(list.files(train_cats_dir)), "\n") #  to check number of images in directors
cat("total training dog images:", length(list.files(train_dogs_dir)), "\n")
cat("total validation cat images:", length(list.files(validation_cats_dir)), "\n")

#### small convent of dogs vs cats classification

library(keras)
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu", input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  layer_conv_2d(filters = 128, kernel_size = c(3,3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  layer_conv_2d(filters = 128, kernel_size = c(3,3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

summary(model) 

## Compiling the model

model %>% compile(
loss = "binary_crossentropy",
optimizer = optimizer_rmsprop(lr = 1e-4),
metrics = c("acc")
)

## Data processing|needs to be in floating point format before fed to the network
## FOUR STEPS
# 1. Read the picture files.
# 2. Decode the JPEG content to RGB grids of pixels.
# 3. Convert these into floating-point tensors.
# 4. Rescale the pixel values (between 0 and 255) to the [0, 1] interval (as you know, neural networks prefer to deal with small input values).

train_datagen <- image_data_generator(rescale = 1/255) # rescale to 1:255
validation_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  train_dir, # target director
  train_datagen, # training data generator 
  target_size = c(150, 150), # resize all images to 150x150
  batch_size = 20, # 20 images 
  class_mode = "binary" # binary_crossentropy loss
)

validation_generator <- flow_images_from_directory(
  validation_dir, # director for validation images
  validation_datagen, # validation data generator
  target_size = c(150, 150), batch_size = 20,
  class_mode = "binary"
)

## example output of a generator
batch <- generator_next(train_generator)
str(batch)

# Fitting the model using a batch generator
history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 30, 
  validation_data = validation_generator,
  validation_steps = 50
)

model %>% save_model_hdf5("cats_and_dogs_small_1.h5") # save the model as a .h5 file
plot(history)

### Using data augmentation
