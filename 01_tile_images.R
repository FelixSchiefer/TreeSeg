

# libraries + path --------------------------------------------------------

pkgs <- c("raster", "rgdal", "rgeos", "foreach", "doParallel", "magick")
sapply(pkgs, require, character.only = TRUE)


workDir = "~/Felix/trees/"
setwd(workDir)
source("00_helper_functions.R")


dataDir <- "/media/sysgen/Volume/Felix/UAVforSAT/TreeSeg/"


## create sampling matrix
set.seed(28)
test  <- sample(1:81, 8)
train <- sample(c(1:81)[-test], 55)
valid <- c(1:81)[-c(test,train)]

m        <- matrix(NA, 9,9)
m[train] <- 1
m[valid] <- 2
m[test]  <- 3


# segment images function -------------------------------------------------

sites <- list.files(dataDir, pattern = "ortho.", recursive = T)
sites <- substr(sites, 13,18)

segmentImages = function(sites, # character
                          outDir, # path to outputfolder
                          useDSM = FALSE,
                          aggregation,
                          tilesize, # numeric, no of pixels (x and y direction) of input raster
                          plot = TRUE,
                          overwrite = FALSE) { # NAvalue, histogram stretch
  
  ## loop over sites
  for(j in 1:length(sites)) {
    site = sites[j]
    
    #### load data ####
    message(paste0("loading data ", j, "/", length(sites)))
    
    ## create outputfolder
    if(useDSM) DSMtag <- "DSM/" else DSMtag <- "noDSM/"
    tileTag <- paste0("t", tilesize)
    resTag  <- paste0("r", aggregation)
    
    imgDir <- paste0(outDir, "img/", site, "/tile", tilesize, "/res", aggregation, "/", DSMtag)
    mskDir <- paste0(outDir, "msk/", site, "/tile", tilesize, "/res", aggregation, "/", DSMtag)
    if(!dir.exists(imgDir)){
      dir.create(imgDir, recursive = TRUE)
      dir.create(mskDir, recursive = TRUE)
    }
    
    ## remove old files if overwrite == TRUE
    if(overwrite) {
      unlink(list.files(imgDir, full.names = TRUE))
      unlink(list.files(mskDir, full.names = TRUE))
    }
    if(length(list.files(imgDir)) > 0 & overwrite == FALSE) {
      stop(paste0("Can't overwrite files in ", imgDir, " -> set 'overwrite = TRUE'"))
    }
    
    ## load area of interest
    AOI <- readOGR(dsn = paste0(dataDir, "shape/ConFoBio_plots.shp"), verbose = FALSE)
    AOI <- gBuffer(AOI, byid = TRUE, width = 0)
    AOI <- AOI[AOI$plot_no == site, ]

    ## load ortho
    orthoFile <- list.files(dataDir, pattern = paste0("ortho_", site, "_res_32N.tif"), recursive = T, full.names = T)
    ortho     <- stack(orthoFile)
    ortho     <- ortho[[-4]] # remove alpha channel
    
    ## crop ortho to AOI
    ortho <- crop(ortho, AOI)
    if(substr(site,1,3) == "HAI") ortho <- mask(ortho, AOI)
    
    ## apply histogram stretch
    q     <- quantile(ortho, probs = c(.001, .999))
    ortho <- (ortho-min(q[,1])) * 255 / (max(q[,2]) - min(q[,1]))
    beginCluster()
    ortho <- clusterR(ortho, fun = reclassify, args = list(rcl = c(-Inf,0,0, 255,Inf,255)), datatype = "INT1U")
    endCluster()
    
    ## set NA values
    if(substr(site,1,3) == "CFB") {
      values(ortho)[values(ortho[[1]]) == 0 & values(ortho[[2]]) == 0 & values(ortho[[3]]) == 0] = NA
    } else {
      values(ortho)[values(ortho[[1]]) == 255 & values(ortho[[2]]) == 255 & values(ortho[[3]]) == 255] = NA
      values(ortho)[values(ortho[[1]]) == 0 & values(ortho[[2]]) == 0 & values(ortho[[3]]) == 0] = NA
    }
      
    ## load DSM
    if(useDSM){
      dsmFile <- list.files(dataDir, pattern = paste0("nDSM_", site, "_res_32N.tif"), full.names = T, recursive = T)
      dsm     <- raster(dsmFile)
      dsm     <- crop(dsm, ortho)
      ortho   <- stack(ortho, dsm)
    }
    
    ## load reference data
    shape <- readOGR(dsn = paste0(dataDir, "shape/poly_", site, ".shp"), verbose = FALSE)
    shape <- gBuffer(shape, byid = TRUE, width = 0)
    shape <- spTransform(shape, crs(ortho))
    
    ## sampling matrix/raster
    bufWidth  <- (100/10.24 - floor(100/10.24))/2 * 10.24
    AOIbuf    <- gBuffer(AOI, width = -bufWidth)
    # shapeCrop <- crop(shape, AOIbuf)
    rasM         <- raster(m, crs = crs(AOIbuf))
    extent(rasM) <- extent(AOIbuf)
    rasShp       <- rasterToPolygons(rasM, dissolve = F)
    
    ## plot site
    if(plot) {
      plotRGB(ortho, colNA = "red")
      lines(shape, lwd = 1.5)
      lines(AOI, col = "red", lwd = 2)
      lines(rasShp, col = "yellow", lwd = 1.5)
    }
    
    #### segment images + masks ####
    message(paste0("segmenting images ", j, "/", length(sites)))
    
    ## define kernel size
    kernelSizeX <- tilesize * xres(ortho)
    kernelSizeY <- tilesize * yres(ortho)
    
    ## create sample positions
    xOffset <- (ncol(ortho)/1024 - floor(ncol(ortho)/1024))/2 * 1024 * xres(ortho)
    yOffset <- (nrow(ortho)/1024 - floor(nrow(ortho)/1024))/2 * 1024 * yres(ortho)
    
    x <- seq(extent(ortho)[1] + kernelSizeX/2 + xOffset,
             extent(ortho)[2] - kernelSizeX/2 - xOffset,
             kernelSizeX)
    y <- seq(extent(ortho)[3] + kernelSizeY/2 + yOffset,
             extent(ortho)[4] - kernelSizeY/2 - yOffset,
             kernelSizeY)
    
    XYpos <- expand.grid(x, y)
    XYpos <- SpatialPointsDataFrame(coords = XYpos, proj4string = crs(AOI), data = XYpos)
    XYpos <- XYpos[AOI,]
    if(plot) points(XYpos, col = "yellow", pch = 3)
    XYpos <- as.data.frame(XYpos)[,c(1,2)]
    XYpos <- cbind(XYpos[,1] - kernelSizeX/2,
                   XYpos[,1] + kernelSizeX/2,
                   XYpos[,2] - kernelSizeY/2,
                   XYpos[,2] + kernelSizeY/2)
    
    
    ## crop images and calc percentage cover of endmember
    cl <- makeCluster(19)
    registerDoParallel(cl)
    
    rmXY <- foreach(i = 1:nrow(XYpos), .packages = c("raster", "rgdal", "keras", "magick"), .combine = "c", .inorder = T) %dopar% {
      
      cropExt <- extent(XYpos[i,])
      
      ## crop and write rasters
      orthoCrop <- crop(ortho, cropExt)
      orthoCrop <- crop(orthoCrop, extent(orthoCrop, 1, tilesize, 1, tilesize)) # remove rounding artifacts
      if(aggregation > 1) orthoCrop <- aggregate(orthoCrop, fact = aggregation)
      
      ## fill NAs in DSM
      NAidx <- which(is.na(values(orthoCrop[[4]])))
      if(length(NAidx) > 0 & length(NAidx) <= 2500) {
        rows <- rowFromCell(orthoCrop, NAidx)
        cols <- colFromCell(orthoCrop, NAidx)
        
        left <- cols-floor(80/2); left[left < 1] = 1
        top  <- rows-floor(80/2); top[top < 1] = 1
        for(k in 1:length(NAidx)) {
          vals <- getValuesBlock(orthoCrop, row = top[k], nrow = 80, col = left[k], ncol = 80, lyrs = 4)
          orthoCrop[[4]][NAidx[k]] = as.numeric(names(table(round(vals))[1]))
        } 
      } else {
        flagPolyNA <- FALSE
      }
      
      ## crop mask
      polyCrop <- crop(shape, cropExt)
      if(length(polyCrop) > 0) { # rasterize shapefile if polygons exist 
        polyCropR <- rasterize(polyCrop, orthoCrop[[1]], field = polyCrop$species_ID)
        polyCropR <- polyCropR - 1 # subtract 1 because Acer spec.
        
        NAidx      <- which(is.na(values(polyCropR)))
        flagPoly0  <- !(0 %in% polyCrop$species_ID) # check if species_ID in data
        flagPolyNA <- length(NAidx) < 2500 # TRUE if NAValues exist AND no less then 2500 (50*50 pixel = 1m2) in crop
        flagOrtho  <- length(which(is.na(values(orthoCrop[[1]])) == TRUE)) == 0 # TRUE if no NA in crop
      } else {
        flagPoly0 <- flagPolyNA <- flagOrtho <- FALSE
      }
      

      if(flagOrtho && flagPoly0 && flagPolyNA) {
        # fill NA values
        if(length(NAidx) > 0) {
          rows <- rowFromCell(polyCropR, NAidx)
          cols <- colFromCell(polyCropR, NAidx)
          
          left <- cols-floor(40/2); left[left < 1] = 1
          top  <- rows-floor(40/2); top[top < 1] = 1
          for(k in 1:length(NAidx)) {
            vals                <- getValuesBlock(polyCropR, row = top[k], nrow = 40, col = left[k], ncol = 40)
            polyCropR[NAidx[k]] <- as.numeric(names(table(vals)[1]))
          } 
        }
        
        extent(orthoCrop) <- extent(0, kernelSizeX, 0, kernelSizeY)
        extent(polyCropR) <- extent(0, kernelSizeX, 0, kernelSizeY)
        
        orthoCrop <- as.array(orthoCrop)
        polyCropR <- as.array(polyCropR)
        orthoCrop <- image_read(orthoCrop / 255)
        polyCropR <- image_read(polyCropR / 255)
        
        filename  <- paste(site, tileTag, resTag, paste0("b", nlayers(ortho)), sep = "_")
        image_write(orthoCrop, format = "png",
                    path = paste0(imgDir, "img", sprintf("%03d", i), "_", filename, ".png"))
        image_write(polyCropR, format = "png",
                    path = paste0(mskDir, "msk", sprintf("%03d", i), "_", filename, ".png"))
      }
      
      # return if tile was exported
      flagOrtho && flagPoly0 && flagPolyNA
    }
    stopCluster(cl)
    
    # export xy positions to a text file
    XYpos           <- as.data.frame(XYpos)
    XYpos           <- XYpos[-which(rmXY == FALSE), ] # remove unexported tiles from list
    colnames(XYpos) <- c("xmin", "xmax", "ymin", "ymax")
    write.csv(XYpos, file = paste0(imgDir, "/metadata_xypos.csv"))
    
    removeTmpFiles(h=0.17)
    gc()
  }
}

segmentImages(sites, outDir = "02_pipeline/softmax/", useDSM = T, tilesize = 1024, aggregation = 2, plot = T, overwrite = T)


