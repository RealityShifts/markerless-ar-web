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
        orb.detectAndCompute(this.frame, mask, kp2, des2);
        mask.delete();
    }

    var dataset = [...Array(des1.rows).keys()].map(function(i) { return Array.from(des1.row(i).data); });
    var index = Flann.fromDataset(dataset, {
        algorithm: Flann.FLANN_INDEX_KDTREE//LSH
    });
    queries = [...Array(des2.rows).keys()].map(function(i) { return Array.from(des2.row(i).data); });
    matches = index.multiQuery(queries, 2);
    index.destroy();

    good = [];
    for(var i = 0; i < matches.length; ++i) {
        var m_n = matches[i];
        var keys = Object.keys(m_n);
        if(keys.length != 2) {
            continue;
        }
        if(Math.abs(m_n[keys[0]] - m_n[keys[1]]) < 7000) {
            good.push({
                query_id: i,
                train_id: m_n[keys[0]] < m_n[keys[1]] ? keys[0] : keys[1]
            });
        }
    }

    var result = false;
    //console.log('good.length: ' + good.length + ', matches.length: ' + matches.length);
    if(good.length >= MIN_MATCH_COUNT) {
        var src_pts = cv.matFromArray(1, 2 * good.length, cv.CV_32FC2, Array.prototype.concat.apply([], good.map(function(m) { var pt = kp1.get(parseInt(m.train_id)).pt; return [pt.x, pt.y]; })));
        var dst_pts = cv.matFromArray(1, 2 * good.length, cv.CV_32FC2, Array.prototype.concat.apply([], good.map(function(m) { var pt = kp2.get(m.query_id).pt; return [pt.x, pt.y]; })));
        /*
        var src_pts = new cv.PointVector();
        var dst_pts = new cv.PointVector();
        for(var m of good) {
            src_pts.push_back(kp1.get(parseInt(m.train_id)).pt);
            dst_pts.push_back(kp2.get(m.query_id).pt);
        }*/
        var mask = new cv.Mat();
        //var homography = cv.findHomography(cv.matFromArray(good.length, 2, cv.CV_32FC2, src_pts), cv.matFromArray(good.length, 2, cv.CV_32FC2, dst_pts), cv.RANSAC, 5.0, mask);
        var homography = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0, mask);

        var size = this.roi.image.size();
        var w = size.width;
        var h = size.height;

        var pts = cv.matFromArray(4, 2, cv.CV_32FC2, [0, 0, 0, h - 1, w - 1, h - 1, w - 1, 0]);
        var corners = new cv.Mat();
        //var homography_withoutc2 = cv.matFromArray(3, 3, cv.CV_32F, homography.data64F);
        cv.perspectiveTransform(pts, corners, homography);
        homography.delete();
        //homography_withoutc2.delete();
        mask.delete();

        if(this.corners != null) {
            this.corners.delete();
            this.src_pts.delete();
            this.dst_pts.delete();
        }
        this.corners = cv.matFromArray(4, 2, cv.CV_32F, corners.data32F.slice(0, 8));
        corners.delete();
        //this.corners = corners;
        this.src_pts = src_pts;
        this.dst_pts = dst_pts;

        result = true;
    }

    kp2.delete();
    des2.delete();

    return result;
};

Matcher.prototype.compute_pose = function(src, dst, rvec, tvec) {
    cv.solvePnP(src, dst, this.camera_matrix, this.dist_coeffs, rvec, tvec);
};
