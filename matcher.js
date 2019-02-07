var Matcher = function(roi, alg, dist_coeffs, camera_matrix) {
    this.roi = roi;
    this.alg = alg;
    this.dist_coeffs = dist_coeffs;
    this.camera_matrix = camera_matrix;

    this.frame = new cv.Mat();
};

MIN_MATCH_COUNT = 50

Matcher.prototype.set_frame = function(frame) {
    cv.cvtColor(frame, this.frame, cv.COLOR_BGR2GRAY);
};

Matcher.prototype.get_correspondence = function() {
    var kp1 = this.roi.keypoints;
    var des1 = this.roi.descriptors;

    if(this.alg === 'orb') {
        var orb = new cv.ORB();
        var kp2 = new cv.KeyPointVector();
        var des2 = new cv.Mat();
        var mask = new cv.Mat();
        orb.detectAndCompute(this.image, mask, kp2, des2);
        mask.delete();
    }

    var dataset = [...Array(des1.rows)].map(function(i) { return des1.row(i).data; });
    var index = Flann.fromDataset(dataset, {
        algorithm: Flann.FLANN_INDEX_LSH
    });
    queries = [...Array(des1.rows)].map(function(i) { return des2.row(i).data; });
    matches = index.multiQuery(queries, 2);

    good = [];
    for(var i = 0; i < matches.length; ++i) {
        var m_n = matches[i];
        var keys = Object.keys(m_n);
        if(keys.length != 2) {
            continue;
        }
        if(Math.abs(m_n[keys[0]] - m_n[keys[1]]) < 0.7) {
            good.push({
                query_id: i,
                train_id: m_n[keys[0]] < m_n[keys[1]] ? keys[0] : keys[1]
            });
        }
    }

    if(good.length >= MIN_MATCH_COUNT) {
    }
};
