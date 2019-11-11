library(tidyverse)
library(keras) # for working with neural networks
install.packages("lime") # for explaining models
library(lime)
install.packages("magick")
library(magick)
library(ggplot2)


## Define parameters to make adapting easy ##
fruit_list <- c("Kiwi", "Banana", "Apricot", "Avocado", "Cocos", "Clementine", "Manderine", "Orange", "Limes", "Lemon", "Peach",
                "Plum", "Raspberry", "Strawberry", "Pineapple", "Pomegranate") #list of fruit to model
output_n <- length(fruit_list) # number of output classes
img_width <- 20 # image size to scale down
img_height <- 20
target_size <- c(img_width, img_height)
channels <- 3 # 3 RGB Channels

#Path to image folders
train_image_files_path <- "/home/liam_local/Documents/Machine_learning/Fruit-Images-Dataset-master/Training"
valid_image_files_path <- "/home/liam_local/Documents/Machine_learning/Fruit-Images-Dataset-master/Validation"

## Data augmentation (optional)
train_data_gen <- image_data_generator(
  rescale = 1/255
)

valid_data_gen <- image_data_generator(
  rescale = 1/255
)

## Load the data
train_image_array_gen <- flow_images_from_directory(train_image_files_path, # reads files from all subfolders
                                                    train_data_gen, target_size = target_size,
                                                    class_mode = "categorical", classes = fruit_list,
                                                    seed = 42)

valid_image_array_gen <- flow_images_from_directory(valid_image_files_path,
                                                    valid_data_gen,
                                                    target_size = target_size,
                                                    classes = fruit_list,
                                                    seed = 42)
cat("Number of images per class:")
table(factor(train_image_array_gen$classes))

cat("\n nClass label vs index mapping: \n")

train_image_array_gen$class_indices

fruits_classes_indices <- train_image_array_gen$class_indices
save(fruits_classes_indices, file =  "/home/liam_local/Documents/Machine_learning/fruits_classes_indices.RData")

#########################

# Define the Keras Model

train_samples <- train_image_array_gen$n # number of training samples

valid_samples <- valid_image_array_gen$n # number of validation samples


# define batch size and number of epochs
batch_size <- 32
epochs <- 10

# Initialise the model #
model <- keras_model_sequential()

# adding layers
model %>% 
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same", input_shape = c(img_width, img_height, channels)) %>%
  layer_activation("relu") %>%
  
  # second hidden layer
  layer_conv_2d(filter = 16, kernel_size = c(3,3), padding = "same") %>%
  layer_activation_leaky_relu(0.5) %>%
  layer_batch_normalization() %>%
  
  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  # Flatten max filtered output into feature vector
  layer_flatten() %>%
  layer_dense(100) %>%
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%
  
  # Outputs from dense layer are projected onto output layer
  layer_dense(output_n) %>%
  layer_activation("softmax")

# compile
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer =  optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
  metrics = "accuracy"
) 

# fit the models
hist <- model %>% fit_generator(
  train_image_array_gen, # training data
  steps_per_epoch = as.integer(train_samples/batch_size),
  epochs = epochs,
  
  validation_data = valid_image_array_gen, # validation data
  validation_steps = as.integer(valid_samples/batch_size),
  
  verbose = 2,
  callbacks = list(
    callback_model_checkpoint("fruits_checkpoinds.h5", save_best_only = TRUE), # save best model after every epoch
    callback_tensorboard(log_dir = "Fruit-Images-Dataset-master/logs")
  )
)
