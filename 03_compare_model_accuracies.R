
# libraries +  path -------------------------------------------------------

pkgs <- c("raster", "doParallel", "foreach", "abind", "tibble")
sapply(pkgs, require, character.only = TRUE)

workDir = "~/Felix/trees/"
setwd(workDir)
source("00_helper_functions.R")


# load data ---------------------------------------------------------------

runs  <- list.files("02_pipeline/softmax/runs/")
epos  <- as.numeric(gsub('^.*_\\s*|\\s*epo.*$', '', runs))
nSpec <- as.numeric(substr(runs, 1, 2))
runs  <- runs[epos == 40 & nSpec == 50]
runs  <- runs[c(3,7,1, 4,8,2, 12,14,6,10)]
sites <- list.files("02_pipeline/softmax/img/")

weights    <- c(2.3887701, 0.2616388, 28.6818629, 0.7869136, 0.9461610, 0.1502391, 0.9411861,
                14.1978020, 11.6882740, 0.8860257, 31.2262007, 3.4815635, 4.1307636, 0.2325988)
invWeights <- 100-weights
# invWeights <- invWeights^2
invWeights <- invWeights/sum(invWeights)
sum(invWeights)


# evaluate models ---------------------------------------------------------

modelAccuracies           <- list()
modelAccuracies$F1perSite <- matrix(data = NA, nrow = length(sites), ncol = length(runs), dimnames = list(sites, runs))
for (i in 1:length(runs)) {
  
  run      <- runs[i]
  
  res      <- as.integer(gsub('^.*_r\\s*|\\s*_.*$', '', run))
  tilesize <- gsub('^.*_t\\s*|\\s*_r.*$', '', run)
  tilesize <- as.integer(as.integer(tilesize)/res)
  dsm      <- as.integer(gsub('^.*_b\\s*|\\s*_.*$', '', run))
  if(dsm == 3) useDSM <- FALSE else useDSM <- TRUE
  
  # load dataset
  checkpointDir <- paste0(workDir, "02_pipeline/softmax/runs/", run, "/checkpoints/")
  load(paste0("02_pipeline/softmax/runs/", run, "/testTiles.RData"))
  # load(paste0("02_pipeline/softmax/runs/", run, "/testTiles.RData"))
  cfb128img <- paste0("02_pipeline/softmax/img/CFB128/tile", tilesize*2, "/res", res, "/DSM/")
  cfb128img <- list.files(cfb128img, pattern = ".png", full.names = T)
  testImg   <- c(testImg, cfb128img)
  
  cfb128msk <- paste0("02_pipeline/softmax/msk/CFB128/tile", tilesize*2, "/res", res, "/DSM/")
  cfb128msk <- list.files(cfb128msk, pattern = ".png", full.names = T)
  testMsk   <- c(testMsk, cfb128msk)
  
  testdata <- tibble(img = testImg,
                     msk = testMsk)
  testDataset <- createDataset(testdata, train = FALSE, batch = 1L, shuffle = FALSE, useDSM = useDSM, tileSize = tilesize)
  
  # load best model
  model = loadModel(checkpointDir, compile = TRUE, custom_objects = list(weightedCategoricalCrossentropy = weightedCategoricalCrossentropy))
  
  
  ##### evaluate model #####
  
  # predict on testdata
  results <- array(data = NA, dim = c(length(testImg), tilesize, tilesize))
  
  pb <- txtProgressBar(0, length(testImg), style = 3)
  datasetIter <- reticulate::as_iterator(testDataset)
  for(j in 1:length(testImg)) {
    img          <- reticulate::iter_next(datasetIter)[[1]]
    pred         <- predict(model, img)
    results[j,,] <- decodeOneHot(pred, progress = F)
    setTxtProgressBar(pb, j)
  }
  
  cl <- makeCluster(19)
  registerDoParallel(cl)
  msks <- foreach(k = 1:dim(results)[1], .inorder = T, .packages = "raster") %dopar% {
    r      <- raster(testMsk[k])
    resRun <- ncol(r)/ncol(results)
    if(resRun > 1) r <- raster::aggregate(r, resRun, fun = max)
    as.array(r)
  }
  stopCluster(cl)
  
  msks <- do.call(abind, list(msks))
  msks <- aperm(msks, c(3,1,2))
  
  predVec <- as.vector(results)+2
  obsVec  <- as.vector(msks)   +1
  u       <- sort(union(obsVec, predVec))
  conmat  <- caret::confusionMatrix(data = factor(predVec, u), reference = factor(obsVec, u))
  
  cmTable   <- conmat$table
  accuracy  <- conmat$overall["Accuracy"]
  kappa     <- conmat$overall["Kappa"]
  precision <- conmat$byClass[,"Precision"]
  recall    <- conmat$byClass[,"Recall"]
  F1Score   <- conmat$byClass[,"F1"]
  
  modelAccuracies$names[i]       <- run
  modelAccuracies$conMat[[i]]    <- cmTable
  modelAccuracies$accuracy[i]    <- accuracy
  modelAccuracies$kappa[i]       <- kappa
  modelAccuracies$precision[[i]] <- precision
  modelAccuracies$recall[[i]]    <- recall
  modelAccuracies$F1Score[[i]]   <- F1Score
  
  site <- unlist(lapply(strsplit(testImg, "_"), "[[", 3))
  for(l in 1:length(unique(site))) {
    idx <- which(site == unique(site)[l])
    
    predVec <- as.vector(results[idx,,])+2
    obsVec  <- as.vector(msks[idx,,])   +1
    u       <- sort(union(obsVec, predVec))
    conmat  <- caret::confusionMatrix(data = factor(predVec, u), reference = factor(obsVec, u))
    
    mIdx <- match(unique(site)[l], rownames(modelAccuracies$F1perSite))
    vals <- if(is.null(nrow(conmat$byClass))) conmat$byClass["F1"] else conmat$byClass[,"F1"] 
    
    modelAccuracies$F1perSite[mIdx, i] <- mean(vals, na.rm = T)
  }
}

save(modelAccuracies, file = "01_code/modelAccuracies_40epo_128_F1pS.RData")
# load("01_code/modelAccuracies_40epo_128_F1pS.RData")



tilesizes  <- as.numeric(gsub('^.*_t\\s*|\\s*_r.*$', '', modelAccuracies$names))
nBands     <- as.numeric(gsub('^.*_b\\s*|\\s*_.*$', '', modelAccuracies$names))
resolution <- as.numeric(gsub('^.*_r\\s*|\\s*_b.*$', '', modelAccuracies$names))
labelNames <- c("AcSp", "CaBe", "FaSy", "FrEx", "QuSp", "TiSp", "Dead", "Soil", "AbAl", "LaDe", "PiAb", "PiSy", "PsMe", "BePe")

resultsTable           <- do.call(cbind, modelAccuracies$F1Score)
resultsTable           <- rbind(resultsTable, modelAccuracies$accuracy, modelAccuracies$kappa)
rownames(resultsTable) <- c(labelNames, "Accuracy", "Kappa")
colnames(resultsTable) <- modelAccuracies$names

write.csv2(modelAccuracies$F1perSite, file = "02_pipeline/softmax/F1perSite.csv")
write.csv2(resultsTable, file = "02_pipeline/softmax/model_evaluation_CFB128.csv")


