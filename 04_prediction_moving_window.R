
# library + path ----------------------------------------------------------

pkgs <- c("raster", "rgdal", "rgeos", "magick", "keras", "caret", "tensorflow", "tibble")
sapply(pkgs, require, character.only = TRUE)

workDir = "~/Felix/trees/"
setwd(workDir)
source("00_helper_functions.R")


# set prerequisites -------------------------------------------------------

weights = c(2.3887701, 0.2616388, 28.6818629, 0.7869136, 0.9461610, 0.1502391, 0.9411861,
            14.1978020, 11.6882740, 0.8860257, 31.2262007, 3.4815635, 4.1307636, 0.2325988)
invWeights = 100-weights
# invWeights = invWeights^2
invWeights = invWeights/sum(invWeights)

modelRuns = list.files("02_pipeline/softmax/runs/")
predModel = modelRuns[4]
checkpointDir = paste0("02_pipeline/softmax/runs/", predModel, "/checkpoints/")
model = loadModel(checkpointDir, compile = TRUE, custom_objects = list(weightedCategoricalCrossentropy = weightedCategoricalCrossentropy))

site = "CFB128"

res      <- as.integer(gsub('^.*_r\\s*|\\s*_.*$', '', predModel))
tilesize <- as.integer(gsub('^.*_t\\s*|\\s*_r.*$', '', predModel))
nPix     <- as.integer(tilesize/res)
nBands   <- as.integer(gsub('^.*_b\\s*|\\s*_.*$', '', predModel))
if(nBands == 4) useDSM <- TRUE else useDSM <- FALSE


# Load data ---------------------------------------------------------------

# load AOI
AOI    <- readOGR("/media/sysgen/Volume/Felix/UAVforSAT/TreeSeg/shape/ConFoBio_plots.shp", verbose = F)
AOI    <- AOI[AOI$plot_no == site, ]
AOIbuf <- gBuffer(AOI, width = tilesize/2/100)

# load ortho
ortho <- stack(paste0("/media/sysgen/Volume/Felix/UAVforSAT/TreeSeg/ortho/", "ortho_", site, "_res_32N.tif"))
ortho <- ortho[[-4]] # remove alpha channel
# plotRGB(ortho); lines(AOI, col = "yellow"); lines(AOIbuf, col = "red")
ortho <- crop(ortho, AOIbuf, datatype = "INT1U")

# apply histogram stretch
q     <- quantile(ortho, probs = c(.001, .999))
ortho <- (ortho-min(q[,1])) * 255 / (max(q[,2]) - min(q[,1]))
ortho <- reclassify(ortho, c(-Inf,0,0, 255,Inf,255), datatype = "INT1U")
ortho <- aggregate(ortho, fact = res)

# load reference shape
shape <- readOGR(paste0("/media/sysgen/Volume/Felix/UAVforSAT/TreeSeg/shape/", "poly_", site, ".shp"), verbose = FALSE)
shape <- gBuffer(shape, byid = TRUE, width = 0)
shape <- spTransform(shape, crs(ortho))

# load DSM
if(useDSM) {
  dsm   <- raster(paste0("/media/sysgen/Volume/Felix/UAVforSAT/TreeSeg/nDSM/", "nDSM_", site, "_res_32N.tif"))
  dsm   <- crop(dsm, ortho)
  dsm   <- resample(dsm, ortho)
  ortho <- stack(ortho, dsm)
}

# plot
plotRGB(ortho, colNA = "red")
lines(AOI, col = "yellow", lwd = 2)
lines(AOIbuf, col = "red", lwd = 2)
lines(shape, lwd = 1.5)



# create tile positions ---------------------------------------------------

nCol    <- ncol(ortho) - 128
nRow    <- nrow(ortho) - 128
xOffset <- round( (nCol/nPix - floor(nCol/nPix))/2 * nPix )
yOffset <- round( (nRow/nPix - floor(nRow/nPix))/2 * nPix )
nTiles  <- 39

# tile positions
idxX <- seq(65+xOffset, nTiles*nPix, nPix)
idxY <- seq(65+yOffset, nTiles*nPix, nPix)

# window shifts
shift <- expand.grid(x = c(-64, 0, 64), y = c(-64, 0, 64))

prediction <- list()
for(p in 1:nrow(shift)) {
  
  idxCol <- idxX + shift[p,"x"]
  idxRow <- idxY + shift[p,"y"]
  idxGrd <- expand.grid(idxRow, idxCol)
  
  # create empty raster
  prediction[[p]] = raster(nrows = nrow(ortho), ncols = ncol(ortho),
                           crs = crs(ortho), ext = extent(ortho), vals = NA)
  
  for(i in 1:nrow(idxGrd)) {
    
    # create tile extent
    xmin       <- idxGrd[i,2]
    xmax       <- idxGrd[i,2] + nPix-1
    ymin       <- idxGrd[i,1]
    ymax       <- idxGrd[i,1] + nPix-1
    cropExtent <- extent(ortho, xmin, xmax, ymin, ymax)
    
    # crop orthomosaic
    orthoCrop  <- crop(ortho, cropExtent)
    
    # prepare image tile for prediction
    orthoCrop <- array_reshape(as.array(orthoCrop/255),
                               dim = c(1, nrow(orthoCrop), ncol(orthoCrop), nBands))
    imgset = orthoCrop %>%
      tf$convert_to_tensor() %>%
      tf$image$convert_image_dtype(dtype = tf$float32) %>%
      tf$reshape(shape = c(1L, nPix, nPix, nBands))
    
    # prediction
    pred <- predict(model, imgset)
    
    # get final class prediction
    predVals <- as.array(k_argmax(pred))[1,,]
    
    # write to raster
    cellIdx                  <- cellsFromExtent(prediction[[p]], cropExtent)
    prediction[[p]][cellIdx] <- t(predVals+2)
    
  }
  # plot(prediction[[p]]); lines(AOI)
}


# stack predictions
predStack <- stack(prediction)

# majority vote
beginCluster()
finalPrediction = clusterR(predStack, calc, args = list(modal, na.rm = T, ties = "random"))
endCluster()

plot(finalPrediction); lines(AOI)

# write to disk
predFile = paste0("/prd_", site, "_t", tilesize, "_r", res, "_b", nBands, "_MV.tif")
writeRaster(finalPrediction, overwrite = TRUE, format= "GTiff",
            filename = paste0("02_pipeline/softmax/predictions/", predModel, predFile),
            options = "COMPRESS=LZW")
