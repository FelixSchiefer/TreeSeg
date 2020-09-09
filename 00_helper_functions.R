
# load keras model --------------------------------------------------------

loadModel = function(path, epoch = NULL, compile = FALSE, custom_objects = NULL) {
  require(keras)
  saved_epochs = as.numeric(gsub(".*weights.(.+)-.*", "\\1", paste0(path, "/", list.files(path))))
  if(is.null(epoch)) { # if no epoch specified load best model...
    loss    = gsub(".*-(.+).hdf5.*", "\\1", paste0(path, "/", list.files(path)))
    loadmod = which(loss == min(loss))[1]
  } else { # else model of specified epoch
    loadmod = which(saved_epochs == epoch)
  }
  # load model
  print(paste0("Loaded model of epoch ", saved_epochs[loadmod], "."))
  load_model_hdf5(paste0(path, "/", list.files(path)[loadmod]), compile = compile, custom_objects = custom_objects)
}


# tfdatasets input pipeline -----------------------------------------------

createDataset <- function(data,
                          train, # logical. TRUE for augmentation of training data
                          batch, # numeric. multiplied by number of available gpus since batches will be split between gpus
                          epochs,
                          useDSM, 
                          shuffle  = TRUE, # logical. default TRUE, set FALSE for test data
                          tileSize = as.integer(tilesize/res),
                          datasetSize) { # numeric. number of samples per epoch the model will be trained on
  require(tfdatasets)
  require(purrr)
  if(useDSM) chnl <- 4L else chnl <- 3L
  
  if(shuffle){
    dataset = data %>%
      tensor_slices_dataset() %>%
      dataset_shuffle(buffer_size = length(data$img), reshuffle_each_iteration = TRUE)
  } else {
    dataset = data %>%
      tensor_slices_dataset() 
  } 
  
  dataset = dataset %>%
    dataset_map(~.x %>% list_modify( # read files and decode png
      img = tf$image$decode_png(tf$io$read_file(.x$img), channels = chnl) %>%
        tf$image$resize(size = c(tileSize, tileSize), method = "nearest"),
      msk = tf$image$decode_png(tf$io$read_file(.x$msk)) %>%
        tf$image$resize(size = c(tileSize, tileSize), method = "nearest")
    )) %>% 
    dataset_map(~.x %>% list_modify(
      img = tf$image$convert_image_dtype(.x$img, dtype = tf$float32), # convert datatype
      msk = tf$subtract(.x$msk, tf$constant(value = 1L, dtype = tf$uint8, shape = c(tileSize,tileSize,1))) %>%
        tf$one_hot(depth = 14L, dtype = tf$uint8) %>% # one-hot encode masks
        tf$squeeze() # removes dimensions of size 1 from the shape of a tensor
    )) %>% 
    dataset_map(~.x %>% list_modify( # set shape to avoid error at fitting stage "tensor of unknown rank"
      img = tf$reshape(.x$img, shape = c(tileSize, tileSize, chnl)),
      msk = tf$reshape(.x$msk, shape = c(tileSize, tileSize, 14L))
    ))
  
  if(train) {
    dataset = dataset %>%
      dataset_map(~.x %>% list_modify( # randomly flip up/down
        img = tf$image$random_flip_up_down(.x$img, seed = 1L),
        msk = tf$image$random_flip_up_down(.x$msk, seed = 1L)
      )) %>%
      dataset_map(~.x %>% list_modify( # randomly flip left/right
        img = tf$image$random_flip_left_right(.x$img, seed = 1L) %>%
          tf$image$random_flip_up_down(seed = 1L),
        msk = tf$image$random_flip_left_right(.x$msk, seed = 1L) %>%
          tf$image$random_flip_up_down(seed = 1L)
      )) %>%
      dataset_map(~.x %>% list_modify( # randomly assign brightness, contrast and saturation to images
        img = tf$image$random_brightness(.x$img, max_delta = 0.1, seed = 1L) %>% 
          tf$image$random_contrast(lower = 0.8, upper = 1.2, seed = 2L) %>%
          # tf$image$random_saturation(lower = 0.8, upper = 1.2, seed = 3L) %>% # requires 3 chnl -> with useDSM chnl = 4 
          tf$clip_by_value(0, 1) # clip the values into [0,1] range.
      )) %>%
      dataset_repeat(count = ceiling(epochs * (datasetSize/length(trainData$img))) )
  }
  
  dataset = dataset %>%
    dataset_batch(batch, drop_remainder = TRUE) %>%
    dataset_map(unname) %>%
    dataset_prefetch(buffer_size = tf$data$experimental$AUTOTUNE)
}


# Subsampling dataset -----------------------------------------------------

dataSplit <- function(probTest = 0.1, probTrain = 0.75, seed = 28, tilesize = tilesize) {
  require(raster)
  
  set.seed(seed)
  m = matrix(NA, 9,9)
  
  nTest  = floor(81*probTest)
  nTrain = ceiling( (81-nTest)*probTrain ) 
  
  test  = sample(1:81, nTest)
  train = sample(c(1:81)[-test], nTrain)
  valid = c(1:81)[-c(test,train)]
  m[train] = 1; m[valid] = 2; m[test]  = 3
  
  rasM = raster(m)
  shpM = rasterToPolygons(rasM, dissolve = F)
  
  if(tilesize == 1024) x = 9 else if(tilesize == 512) x = 18 else x = 36
  mTarget = matrix(1:x^2, x, x)
  rasTarget = raster(mTarget)
  
  idxList = extract(rasTarget, shpM)
  idxTest = unlist(idxList[test])
  idxTrain = unlist(idxList[train])
  idxValid = unlist(idxList[valid])
  
  return(list(test = idxTest, train = idxTrain, valid = idxValid))
}


speciesOccurence = function(data, tilesize = tilesize) {
  out = matrix(NA, nrow = 15, ncol = 3)
  for(j in 1:length(data)) {
    msks = array(NA, dim = c(length(data[[j]]$msk), tilesize, tilesize))
    
    pb = txtProgressBar(min = 0, max = dim(msks)[1], style = 3)
    for(i in 1:dim(msks)[1]) {
      msks[i,,] = as.array(stack(data[[j]]$msk[i]))
      setTxtProgressBar(pb, i)
    }
    t = table(msks)
    out[match(as.numeric(names(t)), 1:14) , j] = t
  }
  return(out)
}


# custom loss function ----------------------------------------------------

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


# decode one-hot encodings ------------------------------------------------

decode_one_hot <- function(x, progress = TRUE) { # TODO: Rewrite using foreach
  require(raster)
  results = array(data = NA, dim = dim(x)[-4])
  if(progress)  pb = txtProgressBar(min = 0, max = dim(results)[1], style = 3)
  for (i in 1:dim(results)[1]) {
    results[i,,] = as.vector(t(which.max(brick(x[i,,,]))))
    if(progress) setTxtProgressBar(pb, i)
  }
  results
}

decodeOneHot <- function(x, progress = TRUE) {
  results = array(data = NA, dim = dim(x)[-4])
  if(progress)  pb = txtProgressBar(min = 0, max = dim(results)[1], style = 3)
  for (i in 1:dim(results)[1]) {
    results[i,,] = as.array(tf$argmax(x[i,,,], axis = 2L))
    if(progress) setTxtProgressBar(pb, i)
  }
  results
}
