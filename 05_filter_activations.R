
# libraries + path ---------------------------------------------------------

pkgs <- c("tensorflow", "keras", "reticulate", "raster")
sapply(pkgs, require, character.only = TRUE)


setwd("~/Felix/trees/")
source("01_code/github/00_helper_functions.R")

weights    <- c(2.3887701, 0.2616388, 28.6818629, 0.7869136, 0.9461610, 0.1502391, 0.9411861,
                14.1978020, 11.6882740, 0.8860257, 31.2262007, 3.4815635, 4.1307636, 0.2325988)
invWeights <- 100-weights
invWeights <- invWeights/sum(invWeights)

outDir <- "50s_t256_r2_b3_40epo"
model  <- loadModel(paste0("02_pipeline/softmax/runs/", outDir, "/checkpoints/"), compile = TRUE,
                    custom_objects = list(weightedCategoricalCrossentropy = weightedCategoricalCrossentropy))


imgPath <- "02_pipeline/softmax/img/CFB056/tile1024/res2/DSM/img014_CFB056_t1024_r2_b4.png"
mskPath <- "02_pipeline/softmax/msk/CFB056/tile1024/res2/DSM/msk014_CFB056_t1024_r2_b4.png"
 

img = tf$image$decode_png(tf$io$read_file(imgPath), channels = 3) %>%
  tf$image$convert_image_dtype(dtype = tf$float32) %>%
  tf$reshape(shape = c(1L, 512L, 512L, 3L))
imgPlot <- as.array(img)[1,,,1:3]

msk     <- tf$image$decode_png(tf$io$read_file(mskPath))
mskPlot <- as.array(msk)[,,1]

# par(mfrow=c(1,1), mar = c(0.1,0.1,0.1,0.1))
# plot(as.raster(imgPlot))

pred       <- predict(model, img)
predDec    <- decode_one_hot(pred)[1,,]
predColIdx <- as.numeric(names(table(as.vector(predDec))))
maskColIdx <- as.numeric(names(table(as.vector(mskPlot))))

# par(mfrow = c(1,3), mar = c(.5,.5,.5,.5))
# plot(as.raster(img_plot, max = 1))
# plot(as.raster(raster(mskPlot),col = viridisLite::viridis(14)[maskColIdx] ))
# plot(as.raster(raster(predDec-1), col = viridisLite::viridis(14)[predColIdx] ))


# filter visualization ----------------------------------------------------

# code based on the following resources:
# https://keras.io/examples/vision/visualizing_what_convnets_learn/#build-a-feature-extraction-model

initializeImage = function(size = size){
  img = tf$random$uniform(shape = c(1L, size, size, 3L))
  (img-0.5) * 0.25
}

deprocessImg = function(img) {
  img = as.array(img)
  dms = dim(img)
  img = img - mean(img)
  img = img / (sd(img) + 1e-5)
  img = img*0.15
  
  img = img + 0.5
  img = pmax(0, pmin(img,1))
  
  array(img, dim = dms[-1])
}

gradientAscentStep = function(img, filterIndex, learningRate) {
  with(tf$GradientTape() %as% tape, {
    tape$watch(img)
    loss = computeLoss(img, filterIndex)
  })
  # compute gradients
  grads = tape$gradient(loss, img)
  # normalize gradients
  grads = tf$math$l2_normalize(grads)
  img = img + learningRate*grads
  return(list(loss, img))
}

visualizeFilter = function(filterIndex, size){
  iterations = 60
  learningRate = 10.0
  img = initializeImage(size)
  
  for(i in 1:iterations) {
    step = gradientAscentStep(img, filterIndex, learningRate)
    loss = step[[1]]
    img = step[[2]]
  }
  deprocessImg(img)
}

generatePattern = function(layerName, filterIndex, size){
  iterations   <- 100
  learningRate <- 10.0
  img          <- initializeImage(size)
  
  layer            <- model$get_layer(layerName)
  featureExtractor <- tf$keras$Model(inputs = model$inputs, outputs = layer$output)
  
  for(i in 1:iterations) {
    
    with(tf$GradientTape() %as% tape, {
      tape$watch(img)
      activation <- featureExtractor(img)
      activation <- activation[,,,filterIndex]
      loss       <- tf$reduce_mean(activation)
    })
    grads <- tape$gradient(loss, img)
    grads <- tf$math$l2_normalize(grads)
    img   <- img + learningRate*grads
  }
  deprocessImg(img)
}


dir.create(paste0("02_pipeline/softmax/runs/", outDir, "/filter100/"))
conv = c(2,5, 9,12, 16,19, 23,26, 30,33, 38,41,44, 49,52,55, 60,63,66, 71,74,77, 80)
for(i in 1:length(conv)) {
  
  imgPerRow <- 16
  size      <- 128L
  
  layerName <- model$layers[[conv[i]]]$name
  noFilters <- dim(model$get_layer(layerName)$output)[[4]]
  if(noFilters > 14) sampSize <- 64 else sampSize <- 14
  nCols     <- sampSize / imgPerRow
  samp      <- sample(noFilters, sampSize, replace = F)
  
  filters   <- list()
  for(j in 1:sampSize) {
    pattern <- generatePattern(layerName, filterIndex = samp[j], size = size)
    filters[[length(filters)+1]] <- pattern
  }
  
  png(paste0("02_pipeline/softmax/runs/", outDir, "/filter100/convolution_", i, "_", layerName, ".png"),
      width  <- imgPerRow*size,
      height <- nCols*size)
  par(mfrow = c(ceiling(nCols), imgPerRow), mar = c(.2,.2,.2,.2))
  for(k in 1:sampSize){
    plot(as.raster(filters[[k]]))
  }
  dev.off()
}