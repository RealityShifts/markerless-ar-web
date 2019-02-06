var ROI = function(image, alg) {
    this.image = new cv.Mat();
    cv.cvtColor(image, this.image, cv.COLOR_BGR2GRAY);

    if(alg === 'orb') {
        orb = cv.ORB();
        this.keypoints = new cv.Mat();
        this.descriptors = new cv.Mat();
        orb.detectAndCompute(this.image, null, this.keypoints, this.descriptors);
    }

   var size = this.image.size();
   var width = size.width;
   var height = size.height;
   var maxSize = width > height ? width : height;
   var w = width / maxSize;
   var h = height / maxSize;

   this.points2d = [[0, 0], [width, 0], [width, height], [0, height]];
   this.points3d = [[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]];
};
