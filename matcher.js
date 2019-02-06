var Matcher = function(roi, alg, dist_coeffs, camera_matrix) {
    this.roi = roi;
    this.alg = alg;
    this.dist_coeffs = dist_coeffs;
    this.camera_matrix = camera_matrix;

    this.frame = new cv.Mat();
};

Matcher.prototype.set_frame = function(frame) {
    cv.cvtColor(frame, this.frame, cv.COLOR_BGR2GRAY);
};

Matcher.prototype.get_correspondence = function() {
    var kp1 = this.roi.keypoints;
    var des1 = this.roi.descriptors;

    if(this.alg === 'orb') {
        var orb = new cv.ORB();
        var kp2 = new cv.Mat();
        var des2 = new cv.Mat();
        orb.detectAndCompute(this.image, null, kp2, des2);
    }

    var dataset = [...Array(des1.rows)].map(function(i) { return des1.row(i).data; });
    var index = Flann.fromDataset(dataset, {
        algorithm: Flann.FLANN_INDEX_LSH
    });
    queries = [...Array(des1.rows)].map(function(i) { return des2.row(i).data; });
    matches = index.multiQuery(queries, 2);
    good = [];
    for(m_n in matches) {
        var keys = Object.keys(m_n);
        if(keys.length != 2) {
            continue;
        }
        if(Math.abs(m_n[keys[0]] - m_n[keys[1]]) < 0.7) {
            good.push(keys[0])
        }
    }
    /*
    flann = new cv.FlannBasedMatcher();
    var matches = new cv.DMatchVectorVector();
    flann.knnMatch(des1, des2, matches);
    */
};
