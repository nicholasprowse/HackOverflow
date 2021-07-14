// Possible improvements:
// Use hysteresis to include weak edges that are attached to strong edges
// Replace Gaussian filter with an adaptive filter to prevent smoothing out edgesToImage

// img has shape [W, H, 3]
function cannyEdgeDetector(img) {
	// First, blur image with 5x5 Gaussian kernel to remove noise
	const blurKernel = tf.tensor([
		[2, 4, 5, 4, 2],
		[4, 9, 12, 9, 4],
		[5, 12, 15, 12, 5],
		[2, 4, 5, 4, 2],
		[4, 9, 12, 9, 4]
	]).div(159);
	let blurred = convolve(img, blurKernel)

	// Calculate gradient along each direction
	const xKernel = tf.tensor([
		[-1, 0, 1],
		[-2, 0, 2],
		[-1, 0, 1]
	]);
	const yKernel = tf.tensor([
		[-1, -2, -1],
		[0, 0, 0],
		[1, 2, 1]
	]);
	let xGrad = convolve(blurred, xKernel)
	let yGrad = convolve(blurred, yKernel)

	// Compute the magnitude and direction of the gradient
	let grad = tf.add(xGrad.square(), yGrad.square()).arraySync()
	let dir = tf.atan2(yGrad, xGrad)
	let rDir = dir.mul(4/Math.PI).round().mul(Math.PI/4).arraySync()
	dir = dir.arraySync()

	// For each pixel, determine if it is an edge.
	// A pixel is an edge if its gradient is above the threshold, and both
	// neighboring pixels along the gradient direction (perpendicular to edge)
	// have a smaller gradient direction
	const h = xGrad.shape[0], w = xGrad.shape[1];
	let mask = tf.buffer([h, w], 'bool')	// To keep track of which edges have been added
	let edges = [];
	const threshold = 0.15;
	for(let c = 0; c < 3; c++) {
		for(let y = 0; y < h; y++) {
		    for(let x = 0; x < w; x++) {
		        if(grad[y][x][c] < threshold || mask.get(y, x))
		            continue;

		        angle = rDir[y][x][c] * Math.PI/180;
		        const dx = Math.round(Math.cos(angle));
		        const dy = Math.round(Math.sin(angle));

		        if(0 <= y+dy && y+dy < h && 0 <= y-dy && y-dy < h)
		            if(0 <= x+dx && x+dx < w && 0 <= x-dx && x-dx < w)
		                if(grad[y+dy][x+dx][c] < grad[y][x][c] && grad[y-dy][x-dx][c] < grad[y][x][c]) {
		                    // We lose 3 pixels on each edge when performing the convolutions, so we add those back in here
							edges.push([x+3, y+3, dir[y][x][c]]);
							mask.set(true, y, x);
						}
		    }
		}
	}

	return edges;
}

/*
 Better idea:
 Store the unrounded gradient direction with each edgel.
 Pick a random edgel. Compute line perpendicular to gradient passing though
 the edgel. Compute the consensus in the same way. If it is below the threshold
 ignore it and try a different random edgel.
 Otherwise refine the line as follows:
 	Take all edgels that agree with the line
	Compute the stdev of the direction of edgels
	Drop any outside of 2 stdev (or some other amount determined empirically)
	Update the line with the mean of the directions of the remaining edgels
	Not sure what the stopping criterion is, since it may be possible that this
	does not converge. Perhaps stop if the line doesn't change or after 5
	updates, which ever comes first?
*/

// Finds n straight line segments from a list of edge pixels
function RANSAC(edges, n) {
	// This algorithm has 3 hyperparameters:
	// d is the maximum distance to a line before an edgel is counted
	// threshReduction is the factor by which the threshold reduces if no lines can be found
	// countThreshold is the number of times to try before reducing the countThreshold
	const d = 2, threshReduction = 0.98, countThreshold = 1000
	lines = []
	let threshold = edges.length / 10
	let count = 0
	while(lines.length < n) {
		let [x, y, a] = edges[Math.floor(Math.random() * edges.length)]
		let m = -1/Math.tan(a)

		let line = [-(m/(m*x - y)), -(1/(-m*x + y)), 1]
		const lineMag = Math.sqrt(line[0]*line[0] + line[1]*line[1])

		// Find all edgels within a distance d of the line
		closeEdges = []
		for(let i = 0; i < edges.length; i++)
			if(Math.abs(edges[i][0]*line[0] + edges[i][1]*line[1] + line[2]) <= d*lineMag)
				closeEdges.push(i)

		// Found a line
		if(closeEdges.length >= threshold) {
			// Add the line, and remove all close edgels so we don't find it again
			lines.push(line);
			for(let i = closeEdges.length - 1; i >= 0; i--)
				edges.splice(closeEdges[i], 1)
			count = 0
		}
		count++

		// If it takes too long, reduce the threshold by threshReduction
		if(count >= countThreshold) {
			threshold = threshReduction * threshold
		}
	}

	return lines;
}

// Scales all elements in a tensor to be in the range 0 to 1 for displaying as image
function normalise(img) {
	img = img.sub(img.min())
	return img.div(img.max())
}

// Tensorflow kernel requires input and output channel dimensions, but ijust
// want to apply the same kernel to all channels independantly, so this function
// generates the appropriate kernel to achieve that.
function convolve(img, kernel) {
	const zero = tf.zerosLike(kernel)
	kernel = tf.stack([
		tf.stack([kernel, zero, zero], 2),
		tf.stack([zero, kernel, zero], 2),
		tf.stack([zero, zero, kernel], 2)
	], 3)
	return tf.conv2d(img, kernel, strides = [1, 1], pad = 'valid')
}

function splitLines(lines) {
	let groups = [[], [], []]
	let count = 0
	do {
		let angles = new Array(lines.length)
		for(let i = 0; i < angles.length; i++) {
			let a = Math.atan2(-lines[i][0], lines[i][1])
			angles[i] = [Math.cos(a), Math.sin(a)]
		}
		let labels = kmeans(angles, 3)
		groups = [[], [], []]
		for(let i = 0; i < lines.length; i++)
			groups[labels[i]].push(lines[i])
		if(count > 50)
			return null
		count++
	} while(groups[0].length != 7 || groups[1].length != 7 || groups[2].length != 7)
	return groups
}

function sortGroup(lines, x, y) {
	let [a, b, c] = lines[0]
	let perp = [-(b/(b*x - a*y)), -(a/(-b*x + a*y)), 1]
	let ref = [(-10000*perp[1] - perp[2])/perp[0], 10000]
	let d = new Array(lines.length)
	for(let i = 0; i < lines.length; i++)
		d[i] = dist(ref, intersection(perp, lines[i]))

	let index = [...Array(lines.length).keys()]
	index.sort((a, b) => d[a] - d[b])
	let newLines = new Array(lines.length)
	for(let i = 0; i < lines.length; i++)
		newLines[i] = lines[index[i]]

	return newLines
}

function getQuadsFromFace(groups, g, start) {
	let others = [0, 1, 2]
	others.splice(g, 1)
	let l = groups[g][start ? 0 : 6]
	let ref = [(-10000*l[1] - l[2])/l[0], 10000]
	let d = new Array(4), p = new Array(4)
	for(let i = 0; i < 4; i++) {
		p[i] = intersection(l, groups[others[Math.floor(i/2)]][6*(i%2)])
		d[i] = dist(ref, p[i])
	}
	let index = [...Array(4).keys()]
	index.sort((a, b) => d[a] - d[b])
	let a = p[index[1]], b = p[index[2]]

	let groupValues = new Array(6).fill(1e10)
	for(let i = 0; i < 3; i++) {
		if(i == g)
			continue

		let ints = [intersection(l, groups[i][0]),
					intersection(l, groups[i][3]),
					intersection(l, groups[i][6])]

		d = new Array(3)
		for(let j = 0; j < 3; j++)
			d[j] = Math.min(dist(ints[j], a), dist(ints[j], b))

		groupValues[2*i] = d[0] + d[1]
		groupValues[2*i+1] = d[1] + d[2]
	}

	let x = tf.tensor(groupValues).argMin().arraySync()
	let g2 = Math.floor(x/2)
	let off1 = start ? 0 : 3
	let off2 = (x % 2) == 0 ? 0 : 3

	let quads = []
	for(let i = 0; i < 3; i++)
		for(let j = 0; j < 3; j++)
			quads.push([groups[g][i+off1], groups[g2][j+off2],
					    groups[g][i+off1+1], groups[g2][j+off2+1]])
	return quads
}

function kmeans(data, k) {
    let n = data.length
	for(let i = 0; i < n; i++)
		data[i] = tf.tensor(data[i])

	let means = new Array(k)
	for(let i = 0; i < k; i++)
		means[i] = data[Math.floor(Math.random() * n)]

	while(true) {
	    // Create label image containing which mean each pixel is closest to
	    label = new Array(n)
	    for(let i = 0; i < n; i++) {
			let min = 0, minDist = 1e10
			for(let j = 0; j < k; j++) {
		        let d = data[i].squaredDifference(means[j]).sum().arraySync()

		        if(d < minDist){
					minDist = d
					min = j
				}
			}
			label[i] = min
	    }

	    // Recompute means using the new labels
	    new_means = new Array(k)
	    total = new Array(k).fill(0)
		for(let i = 0; i < k; i++)
			new_means[i] = tf.zerosLike(data[0])

	    for(let i = 0; i < n; i++) {
	        new_means[label[i]] = new_means[label[i]].add(data[i])
	        total[label[i]]++
	    }

		let finished = true;
		for(let i = 0; i < k; i++) {
	    	new_means[i] = new_means[i].div(total[i])
			if(new_means[i].notEqual(means[i]).sum().arraySync()[0])
				finished = false
			means[i] = new_means[i]
		}

		if(finished) {
			for(let i = 0; i < n; i++)
				data[i] = data[i].arraySync()
			return label
		}
	}
}

function dist(a, b){
	let sum = 0
	for(let i = 0; i < a.length; i++)
		sum += (a[i] - b[i])**2
	return sum
}

function intersection(a, b) {
	let [x, y, l] = cross(a, b)
	return [x/l, y/l]
}

function cross(a, b) {
	let [x, y, z] = a, [u, v, w] = b
	return [w*y - v*z, -w*x + u*z, v*x - u*y]
}

//************ THESE FUNCTIONS ARE ALL FOR TESTING PURPOSES ONLY ***************

function drawEdges(img, edges) {
	img = img.bufferSync();
	for(let i = 0; i < edges.length; i++) {
		img.set(1, edges[i][1], edges[i][0], 0)
		img.set(0, edges[i][1], edges[i][0], 1)
		img.set(0, edges[i][1], edges[i][0], 2)
	}
	return img.toTensor()
}

// Warning: Divide by zero errors are not handled
function drawLine(img, line, c) {
	const m = -line[0] / line[1];
	if(Math.abs(m) > 1) {
		// Iterate over y
		for(let y = 0; y < img.shape[0]; y++) {
			const x = -Math.round((line[1]*y + line[2]) / line[0])
			if(x >= 0 && x < img.shape[1]) {
				img.set(c[0], y, x, 0)
				img.set(c[1], y, x, 1)
				img.set(c[2], y, x, 2)
			}
		}
	} else {
		// Iterate over x
		for(let x = 0; x < img.shape[1]; x++) {
			const y = -Math.round((line[0]*x + line[2]) / line[1])
			if(y >= 0 && y < img.shape[0]) {
				img.set(c[0], y, x, 0)
				img.set(c[1], y, x, 1)
				img.set(c[2], y, x, 2)
			}
		}
	}

	return img
}

function drawLineSegment(img, p1, p2, c) {
	let line = cross([p1[0], p1[1], 1], [p2[0], p2[1], 1])
	const m = -line[0] / line[1];
	if(Math.abs(m) > 1) {
		// Iterate over y
		let sy = Math.min(p1[1], p2[1]), fy = Math.max(p1[1], p2[1])
		for(let y = Math.floor(sy); y <= fy; y++) {
			const x = -Math.round((line[1]*y + line[2]) / line[0])
			if(x >= 0 && x < img.shape[1]) {
				img.set(c[0], y, x, 0)
				img.set(c[1], y, x, 1)
				img.set(c[2], y, x, 2)
			}
		}
	} else {
		// Iterate over x
		let sx = Math.min(p1[0], p2[0]), fx = Math.max(p1[0], p2[0])
		for(let x = Math.floor(sx); x <= fx; x++) {
			const y = -Math.round((line[0]*x + line[2]) / line[1])
			if(y >= 0 && y < img.shape[0]) {
				img.set(c[0], y, x, 0)
				img.set(c[1], y, x, 1)
				img.set(c[2], y, x, 2)
			}
		}
	}

	return img
}

function drawQuad(img, q) {
	let corners = new Array(4)
	for(let i = 0; i < 4; i++)
		corners[i] = intersection(q[i], q[(i+1) % 4])

	for(let i = 0; i < 4; i++)
		img = drawLineSegment(img, corners[i], corners[(i+1)%4], [0, 0, 0])
	return img
}

function test() {
	const im = new Image();
	im.onload = () => {
		let img = tf.browser.fromPixels(im)
		// Want img to have maximum size of 500
		scale = 500 / Math.max(img.shape[0], img.shape[1])
		img = tf.image.resizeBilinear(img, [img.shape[0] * scale, img.shape[1] * scale]).div(255)

		let edges = cannyEdgeDetector(img)
		let lines = RANSAC(edges, 21)
		let groups = splitLines(lines)
		if(groups == null) {
			console.log("FAILURE")
			return
		}
		for(let i = 0; i < 3; i++)
			groups[i] = sortGroup(groups[i], img.shape[1]/2, img.shape[0]/2)

		let quads = getQuadsFromFace(groups, 0, false)
		quads = quads.concat(getQuadsFromFace(groups, 0, true))
		let s = ""
		for(let g of groups[0])
			s += "{" +g[0] + "," + g[1] + "},"
		//console.log(s)

		img = img.bufferSync()
		// img = overlayEdges(a, edges)
		for(let i = 0; i < groups[0].length; i++)
			img = drawLine(img, groups[0][i], [1, 0, 0])
		for(let i = 0; i < groups[1].length; i++)
			img = drawLine(img, groups[1][i], [0, 1, 0])
		for(let i = 0; i < groups[2].length; i++)
			img = drawLine(img, groups[2][i], [0, 0, 1])
		for(let i = 0; i < quads.length; i++)
			img = drawQuad(img, quads[i])

		showImg(img.toTensor());
	}
	im.src = "../imgs/cube_1.jpeg";
}

function showImg(img) {
	img = normalise(img);
	let canvas = document.createElement('canvas');
	document.body.insertBefore(canvas, null);
	tf.browser.toPixels(img, canvas);
}
