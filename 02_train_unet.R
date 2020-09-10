
# libraries + path -----------------------------------------------------------

pkgs <- c("keras", "tidyverse", "tibble", "tensorflow")
sapply(pkgs, require, character.only = TRUE)

tf$compat$v1$set_random_seed(as.integer(28))

# when runing multi gpu model
# strategy <- tf$distribute$MirroredStrategy()
# strategy$num_replicas_in_sync

workDir = "~/Felix/trees/"
setwd(workDir)
source("00_helper_functions.R")


dataDir <- "/media/sysgen/Volume/Felix/UAVforSAT/TreeSeg/"


# Parameters --------------------------------------------------------------

tilesize   <- 256L
res        <- 2
useDSM     <- TRUE
noEpochs   <- 20
repeatData <- 4
if(useDSM) noBands <- 4 else noBands <- 3 # if ortho is combined with DSM = 4 (RGB + DSM) else DSM = 3 (RGB)

excludeSites <- c("CFB128")
# excludeSites <- NULL
nSites       <- length(list.files(dataDir, pattern = "ortho.", recursive = T))
nSites       <- nSites - length(excludeSites)

siteTag <- paste0(nSites, "s")
tileTag <- paste0("t", tilesize)
resTag  <- paste0("r", res)
epoTag  <- paste0(noEpochs, "epo")
DSMTag  <- paste0("b", noBands)
# if(useDSM) DSMTag <- "b4" else DSMTag = "b3"

# outDir = "34sCFB_512_2cm_tfdataset_DSM_100epo/"
outDir = paste(siteTag, tileTag, resTag, DSMTag, epoTag, "WCCE2", sep = "_")
dir.create(paste0("02_pipeline/softmax/runs/", outDir), recursive = TRUE)


# Load Data ---------------------------------------------------------------

## list all data
pathPattern <- paste(tileTag, "r2", sep = "_")
pathImg     <- list.files("~/Felix/trees/02_pipeline/softmax/img/", full.names = T, pattern = pathPattern, recursive = T)
pathMsk     <- list.files("~/Felix/trees/02_pipeline/softmax/msk/", full.names = T, pattern = pathPattern, recursive = T)
## exclude sites
if(!is.null(excludeSites)) {
  excludeIdx <- which(substr(pathImg, 51, 56) %in% excludeSites)
  pathImg    <- pathImg[-excludeIdx]
  pathMsk    <- pathMsk[-excludeIdx]
}



# Data split --------------------------------------------------------------

## split test-/train-/valid-data
idx         <- dataSplit(probTest = 0.1, probTrain = 0.75, seed = 28, tilesize = tilesize)
tileNumbers <- as.numeric(gsub('^.*DSM/img\\s*|\\s*_.*$', '', pathImg))

testImg  <- pathImg[tileNumbers %in% idx$test]
testMsk  <- pathMsk[tileNumbers %in% idx$test]
trainImg <- pathImg[tileNumbers %in% idx$train]
trainMsk <- pathMsk[tileNumbers %in% idx$train]
validImg <- pathImg[tileNumbers %in% idx$valid]
validMsk <- pathMsk[tileNumbers %in% idx$valid]

## save test data to disk
save(testImg, testMsk, file = paste0("02_pipeline/softmax/runs/", outDir, "/testTiles.RData"), overwrite = T)

## create tibbles
testData  <- tibble(img = testImg, msk = testMsk)
trainData <- tibble(img = trainImg, msk = trainMsk)
validData <- tibble(img = validImg, msk = validMsk)

## calculate dataset size
datasetSize <- length(trainData$img) * repeatData

# # calculate area-related share of species in dataset
# areaShare = speciesOccurence(data = list(testData, trainData, validData), tilesize = tilesize)
# areaShare[15,] = colSums(areaShare, na.rm = T)
# areaSharePer = apply(areaShare, 2, FUN = function(x) x[-15]/x[15])
# colnames(areaShare) = colnames(areaSharePer) = c("test", "train", "valid")
# round(rowSums(areaShare)[-15]/rowSums(areaShare)[15]*100, 2)
# save(areaShare, areaSharePer, file = "01_code/areaShare_seed28_10_75_25_50s.RData")
# load("01_code/areaShare_seed28_10_75_25_50s.RData")


# Create datasets ----------------------------------------------------------------

## define batch size
if(tilesize == 1024) batchSize <- 3 else if(tilesize == 512) batchSize <- 12 else batchSize <- 46 # (multi gpu, 512 a 2cm --> rstudio freeze) 3, 12, 46

trainingDataset   <- createDataset(trainData, train = T, batch = batchSize, epochs = noEpochs, datasetSize = datasetSize, useDSM = useDSM)
validationDataset <- createDataset(validData, train = F, batch = batchSize, epochs = noEpochs, useDSM = useDSM)


# Custom loss function ----------------------------------------------------

weights    <- c(2.3887701, 0.2616388, 28.6818629, 0.7869136, 0.9461610, 0.1502391, 0.9411861,
                14.1978020, 11.6882740, 0.8860257, 31.2262007, 3.4815635, 4.1307636, 0.2325988)
invWeights <- 100-weights
# invWeights <- invWeights^2
invWeights <- invWeights/sum(invWeights)
sum(invWeights)

weightedCategoricalCrossentropy <- function(yTrue, yPred, weights = invWeights) {
  
  # code based on the following resources:
  # https://keras.rstudio.com/articles/examples/unet.html
  # https://stackoverflow.com/questions/51316307/custom-loss-function-in-r-keras
  
  kWeights <- k_constant(weights, dtype = tf$float32, shape = c(1,1,1,14))
  
  yWeights <- kWeights * yPred
  yWeights <- k_sum(yWeights, axis = 4L)

  loss     <- tf$keras$losses$categorical_crossentropy(yTrue, yPred)
  wLoss    <- yWeights * loss
  
  return(tf$reduce_mean(wLoss))
}
# weightedCategoricalCrossentropy(yTrue, yPred, invWeights)

wcce_loss <- function(yTrue, yPred) {
  result <- weightedCategoricalCrossentropy(yTrue, yPred, weights = invWeights)
  return(result)
}
# wcce_loss(yTrue, yPred)


# Define U-net CNN --------------------------------------------------------

# U-net code is based on the following github example:
# https://github.com/rstudio/keras/blob/master/vignettes/examples/unet_linux.R

getUnet <- function(inputShape = c(as.integer(tilesize/res), as.integer(tilesize/res), noBands),
                    numClasses = 14) {
  
  # create blocks
  inputs <- layer_input(shape = inputShape)

  down1 <- inputs %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") 
  down1_pool <- down1 %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))

  down2 <- down1_pool %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") 
  down2_pool <- down2 %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))

  down3 <- down2_pool %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") 
  down3_pool <- down3 %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))

  down4 <- down3_pool %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") 
  down4_pool <- down4 %>%
    layer_max_pooling_2d(pool_size = c(2, 2), strides = c(2, 2))

  center <- down4_pool %>%
    layer_conv_2d(filters = 1024, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 1024, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") 

  up4 <- center %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    {layer_concatenate(inputs = list(down4, .), axis = 3)} %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 512, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")

  up3 <- up4 %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    {layer_concatenate(inputs = list(down3, .), axis = 3)} %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 256, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")

  up2 <- up3 %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    {layer_concatenate(inputs = list(down2, .), axis = 3)} %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")

  up1 <- up2 %>%
    layer_upsampling_2d(size = c(2, 2)) %>%
    {layer_concatenate(inputs = list(down1, .), axis = 3)} %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same") %>%
    layer_batch_normalization() %>%
    layer_activation("relu")

  classify <- layer_conv_2d(up1,
                            filters = numClasses, 
                            kernel_size = c(1, 1),
                            activation = "softmax")
  
  # build model
  model <- keras_model(
    inputs = inputs,
    outputs = classify
  )
  
  model %>% compile(
    optimizer = tf$keras$optimizers$RMSprop(0.0001),
    # loss = "categorical_crossentropy",
    loss = weightedCategoricalCrossentropy,
    metrics = c("accuracy", "categorical_crossentropy")
    # metrics = custom_metric("wcce", wcce_loss)
  )
  
  return(model)
}

# multiple gpu (custom loss/metric not supported)
with(strategy$scope(), {
  model <- getUnet()
})

# single gpu
model <- getUnet()



# Train U-net -------------------------------------------------------------

checkpointDir <- paste0(workDir, "02_pipeline/softmax/runs/", outDir, "/checkpoints/")
# unlink(checkpointDir, recursive = TRUE)
dir.create(checkpointDir, recursive = TRUE)
filepath <- file.path(checkpointDir, "weights.{epoch:02d}-{val_loss:.5f}.hdf5")

cp_callback <- callback_model_checkpoint(filepath = filepath,
                                         monitor = "val_loss",
                                         save_weights_only = FALSE,
                                         save_best_only = FALSE,
                                         verbose = 1,
                                         mode = "auto",
                                         save_freq = "epoch")

history <- model %>% fit(x = trainingDataset,
                         epochs = noEpochs,
                         steps_per_epoch = datasetSize/(batchSize),
                         callbacks = list(cp_callback,
                                          callback_terminate_on_naan()),
                         validation_data = validationDataset)



# Model evaluation --------------------------------------------------------

plot(history)

# checkpointDir <- paste0(workDir, "02_pipeline/softmax/runs/", outDir, "/checkpoints/")
# load(paste0("02_pipeline/softmax/runs/", outDir, "/testTiles.RData"))
# testData <- tibble(img = test_img,msk = test_msk)
testDataset <- createDataset(testData, train = FALSE, batch = 1, shuffle = FALSE, useDSM = TRUE)

# model <- loadModel(checkpointDir, compile = TRUE)
model <- loadModel(checkpointDir, compile = TRUE, custom_objects = list(weightedCategoricalCrossentropy = weightedCategoricalCrossentropy))

eval  <- evaluate(object = model, x = testDataset)
eval


