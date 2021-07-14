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

		        angle = rDir[y][x][c];
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

function brightenColor(color) {
	let [r, g, b] = color
	let max = Math.max(r, g, b), min = Math.min(r, g, b)
	let delta = max - min
	let h = 0
	if(max == r)
		h = ((g - b) / delta) % 6
	if(max = g)
		h = ((b - r) / delta) + 2
	if(max = b)
		h = ((r - g) / delta) + 4
	h *= 60
	let s = 0
	if(max > 0)
		s = delta / max


	let x = s * (1 - Math.abs((h/60) % 2 - 1))
	if(h < 60)
		[r, g, b] = [s, x, 0]
	else if(h < 120)
		[r, g, b] = [x, s, 0]
	else if(h < 180)
		[r, g, b] = [0, s, x]
	else if(h < 240)
		[r, g, b] = [0, x, s]
	else if(h < 300)
		[r, g, b] = [x, 0, s]
	else
		[r, g, b] = [s, 0, x]
	return [r, g, b]
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
	let threshold = edges.length / 15
	let count = 0
	while(lines.length < n) {
		// Select random edgel and define line passing through edgel
		// perpedicular to the edgels gradient
		const [x, y, a] = edges[Math.floor(Math.random() * edges.length)]
		const m = -1/Math.tan(a)		// Gradient of line
		const line = [m/(y-m*x), -1/(y-m*x), 1]
		const lineMag = Math.sqrt(line[0]*line[0] + line[1]*line[1])

		// Find all edgels within a distance d of the line
		closeEdges = []
		for(let i = 0; i < edges.length; i++)
			if(Math.abs(edges[i][0]*line[0] + edges[i][1]*line[1] + line[2]) <= d*lineMag)
				closeEdges.push(i)

		// // Refine the line
		// const angles = tf.buffer([closeEdges.length])
		// let idx = 0
		// for(let i of closeEdges)
		// 	angles.set(closeEdges[idx++], edges[i][2])
		// const moments = angles.toTensor().moments().variance
		// let min = moments.mean - 2*Math.sqrt(moments.variance)
		// let max = moments.mean + 2*Math.sqrt(moments.variance)
		// if(min < 0) min += 2*Math.PI
		// if(max > Math.PI) max -= Math.PI
		// for(let i = closeEdges.length - 1; i >= 0; i--)
		// 	if(edges[closeEdges[i]] < min)

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
		if(count >= countThreshold)
			threshold = threshReduction * threshold

	}
	return lines;
}

/*
 Takes in the list of lines generated from RANSAC, and the size of the image (as
 generated by img.shape)
 Basic Idea: For each pair of lines, compute the intersection point
 Store the intersection point with along with both lines if and only if the
 intersection is outside of the bounds of the image. After some experimentation
 I have found that it is possible but very unlikely for the vanishing points to
 be inside the image assuming the cube takes up most of the image
 Use K-means to cluster these intersections into 3 clusters
 Within each cluster, remove outliers
 Now, within each cluster, all the lines remaining are likely to be valid, as
 they all intersect at a similar location (i.e. they are parellel in 3D space)
*/

function removeInvalidLines(lines, size){
	// Get all intersections outside of the image
	let angles = [], intLines = []
	for(let i = 0; i < lines.length; i++)
		for(let j = i+1; j < lines.length; j++) {
			const [a, b, c] = lines[i],
			      [d, e, f] = lines[j]
			let [x, y] = [(b*f - c*e) / (a*e - b*d), (c*d - a*f) / (a*e - b*d)]
			if(x < 0 || y < 0 || x > size[1] || y > size[0]) {
				angles.push(Math.atan2(y - size[0], x - size[1]))
				intLines.push([i, j])
			}
		}

	// let s = "{"
	// for(let i of angles){
	// 	s += i + ","
	// }
	// console.log(s)
	// console.log(intLines)
	// Group the intersections
	// let means = [tf.tensor([size[1]/2, size[0]]), tf.tensor([0, 0]), tf.tensor([size[1], 0])]
	let labels = kmeans(angles, 3)

	// console.log(angles)
	// console.log(labels)

	let groups = [[], [], []]
	let lineGroups = [[], [], []]
	for(let i = 0; i < angles.length; i++) {
		groups[labels[i]].push(angles[i])
		lineGroups[labels[i]].push(intLines[i])
	}
	//console.log(groups)
	// Remove outliers (more than 2 stdevs from mean of each group)
	means = []
	for(let i = 0; i < 3; i++) {
		let s = "{"
		for(let a of angles){
			s += a + ","
		}
		console.log(s)
		let moments = tf.moments(tf.tensor(groups[i]))
		let mean = moments.mean.dataSync()[0]
		let std = moments.variance.sqrt().dataSync()[0]
		for(let j = groups[i].length-1; j >= 0; j--) {
			if(Math.abs(mean - groups[i][j]) > 2*std) {
				groups[i].splice(j, 1)
				lineGroups[i].splice(j, 1)
			}
		}
		console.log(groups[i])
		means.push(tf.moments(tf.tensor(groups[i])).mean.dataSync()[0])
	}
	console.log(lineGroups)
	// Extract lines from group. Each line is assigned to the group it's
	// intersection is closest to the mean of
	labels = new Array(lines.length).fill(-1)
	closest = new Array(lines.length).fill(-1)
	for(let i = 0; i < 3; i++) {
		for(let j = 0; j < lineGroups[i].length; j++) {
			// Distance from angle to mean
			let d = Math.abs(groups[i][j] - means[i])
			if(closest[j] == -1 || d < closest[j]) {
				labels[lineGroups[i][j][0]] = i
				labels[lineGroups[i][j][1]] = i
				closest[j] = d
			}
		}
	}

	return lines, labels
}

function segmentLines(lines) {
	let angles = []
	for(let l of lines) {
		let a = Math.atan2(l[0], -l[1])
		angles.push([Math.cos(a), Math.sin(a)])
	}

	do {
		labels = kmeans(angles, 3)

		// Sort lines into groups
		groups = [[], [], []]
		for(let i = 0; i < lines.length; i++)
			groups[labels[i]].push(lines[i])

	} while(groups[0].length != 7 || groups[1].length != 7 || groups[2].length != 7)

	return groups
}

function kmeans(data, k, means=null) {
	let n = data.length
	for(let i = 0; i < n; i++)
		data[i] = tf.tensor(data[i])

	if(means == null) {
		means = new Array(k)
		for(let i = 0; i < k; i++)
			means[i] = data[Math.floor(Math.random() * data.length)]
	}

	while(true) {
	    // Create label array containing which mean each pixel is closest to
	    let label = new Array(n);
	    for(let i = 0; i < n; i++) {
			let min = 0, minDist = 1e10
			for(let j = 0; j < k; j++) {
				const dist = data[i].squaredDifference(means[j]).sum().dataSync()[0]
		        if(dist < minDist){
					minDist = dist
					min = j
				}
			}
			label[i] = min
	    }

	    // Recompute means using the new labels
	    new_means = new Array(k)
	    total = new Array(k)
		for(let i = 0; i < k; i++){
			new_means[i] = tf.zerosLike(data[0])
			total[i] = 0
		}
	    for(let i = 0; i < n; i++) {
	        new_means[label[i]] = new_means[label[i]].add(data[i])
	        total[label[i]]++
	    }

		let finished = true
		for(let i = 0; i < k; i++) {
			new_means[i] = new_means[i].div(total[i])
			if(new_means[i].notEqual(means[i]).sum().dataSync()[0])
				finished = false
			means[i] = new_means[i]
		}

		if(finished) {
			for(let i = 0; i < n; i++)
				data[i] = data[i].dataSync()
			return label
		}
	}
}

function getQuads(img, groups) {
	// Sort each group so the lines are in order
	for(let i = 0; i < 3; i++) {
		let [x, y] = [img.shape[1]/2, img.shape[0]/2]
		let m = groups[i][0][1] / groups[i][0][0]
		let perp = [m/(y - m*x), -1/(y - m*x), 1]
		let ref = [(-10000*perp[1] - perp[2])/perp[0], 10000]
		let d = new Array(groups[i].length)
		for(let j = 0; j < groups[i].length; j++)
			d[j] = dist(ref, intersection(perp, groups[i][j]))
		let index = [...Array(groups[i].length).keys()]
		index.sort((a, b) => (d[a] - d[b]))
		let newLines = new Array(groups[i].length)
		for(let j = 0; j < groups[i].length; j++)
			newLines[j] = groups[i][index[j]]
		groups[i] = newLines
	}

	let quads = []

	quads = quads.concat(getQuadsFromFace(groups, 1, false))
	quads = quads.concat(getQuadsFromFace(groups, 1, true))

	return quads
}

/*
 Given 3 groups of lines, this will return the quads on the face attached to the
 given line. group is the index of the group that line belongs to, and start
 determines if the line of interest is at the start or end of the group
*/
function getQuadsFromFace(lines, group, start) {
	let quads = []
	// We can determine the extent of
	// the line by determining the region where it is between both other groups
	let l1 = lines[group][start ? 0 : 6]
	let ref = [(-10000*l1[1] - l1[2])/l1[0], 10000]
	let d = new Array(4)
	let ints = new Array(4)
	for(let i = 0; i < 4; i++) {
		let l2 = lines[Math.floor(i/2)][6*(i % 2)]
		ints[i] = intersection(l1, l2)
		d[i] = (ints[i][0] - ref[0])**2 + (ints[i][1] - ref[1])**2
	}
	let index = [...Array(4).keys()]
	index.sort((a, b) => (d[a] - d[b]))
	let a = ints[index[1]], b = ints[index[0]]

	// The group that has lines intersect at both a and b is the group that
	// corresponds to this side of the first group
	let groupValues = new Array(6).fill(1e10)
	for(let g = 0; g < 3; g++) {
		if(g == group)
			continue

		let ints = [intersection(l1, groups[g][0]),
					intersection(l1, groups[g][3]),
					intersection(l1, groups[g][6])]

		let d = new Array(3)
		for(let i = 0; i < 3; i++)
			d[i] = Math.min(dist(ints[i], a), dist(ints[i], b))

		groupValues[2*g] = d[0] + d[1]
		groupValues[2*g+1] = d[1] + d[2]
	}
	console.log(groupValues)
	let x = tf.tensor(groupValues).argMin().arraySync()
	let group2 = Math.floor(x / 2)
	console.log(group2)
	let off1 = start ? 0 : 3
	let off2 = (x % 2) == 1 ? 0 : 3
	for(let i = 0; i < 3; i++) {
		for(let j = 0; j < 3; j++) {
			quads.push([
				lines[group][i+off1], lines[group2][j+off2],
				lines[group][i+off1+1], lines[group2][j+off2+1]])
		}
	}

	return quads
}

function dist(p1, p2) {
	return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
}

function intersection(l1, l2) {
	let int = cross(l1, l2)
	return [int[0]/int[2], int[1]/int[2]]
}

// Determines if p is inside quad defined by 4 lines
// l1 and l3 should be on opposite sides of quad
function inQuad(p, l1, l2, l3, l4) {
	let p1 = cross(l1, l2),
		p2 = cross(l2, l3),
		p3 = cross(l3, l4),
		p4 = cross(l4, l1)

	p1 = [p1[0]/p1[2], p1[1]/p1[2]]
	p2 = [p2[0]/p2[2], p2[1]/p2[2]]
	p3 = [p3[0]/p3[2], p3[1]/p3[2]]
	p4 = [p4[0]/p4[2], p4[1]/p4[2]]

	return inTriangle(p, p1, p2, p3) || inTriangle(p, p2, p3, p4)
}

// Determines if p is inside triangle defined by 3 corner points
function inTriangle(p, a, b, c) {
	// Compute vectors
	let v0 = [c[0] - a[0], c[1] - a[1]],
		v1 = [b[0] - a[0], b[1] - a[1]],
		v2 = [p[0] - a[0], p[1] - a[1]]

	// Compute dot products
	let dot00 = dot(v0, v0),
		dot01 = dot(v0, v1),
		dot02 = dot(v0, v2),
		dot11 = dot(v1, v1),
		dot12 = dot(v1, v2)

	// Compute barycentric coordinates
	let invDenom = 1 / (dot00 * dot11 - dot01 * dot01),
		u = (dot11 * dot02 - dot01 * dot12) * invDenom,
		v = (dot00 * dot12 - dot01 * dot02) * invDenom

	// Check if point is in triangle
	return (u > 0) && (v > 0) && (u + v < 1)
}

function cross(a, b) {
	let [x, y, z] = a, [u, v, w] = b
	return [-w*y + v*z, w*x - u*z, -v*x + u*y]
}

function dot(a, b) {
	let sum = 0;
	for(let i = 0; i < a.length; i++)
		sum += a[i] * b[i];
	return sum;
}

// Scales all elements in a tensor to be in the range 0 to 1 for displaying as image
function normalise(img) {
	img = img.sub(img.min())
	return img.div(img.max())
}

// Tensorflow kernel requires input and output channel dimensions, but I just
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
	return img
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

function overlayQuad(img, q) {
	let intersections = new Array(4)
	for(let i = 0; i < 4; i++)
		intersections[i] = intersection(q[i], q[(i+1) % 4])
	console.log(intersections, q)
	for(let i = 0; i < 4; i++)
		img = overlayLineSegment(img, intersections[i], intersections[(i+1) % 4])

	return img
}

function overlayLineSegment(img, p1, p2) {
	let line = [p1[1] - p2[1], p2[0] - p1[0], p1[0]*p2[1] - p1[1]*p2[0]]
	const m = -line[0] / line[1];
	if(Math.abs(m) > 1) {
		// Iterate over y
		let sy = Math.min(p1[1], p2[1]), fy = Math.max(p1[1], p2[1])

		for(let y = Math.floor(sy); y <= fy; y++) {
			const x = -Math.round((line[1]*y + line[2]) / line[0])
			if(x >= 0 && x < img.shape[1]) {
				img.set(0, y, x, 0)
				img.set(0, y, x, 1)
				img.set(0, y, x, 2)
			}
		}
	} else {
		// Iterate over x
		let sx = Math.min(p1[0], p2[0]), fx = Math.max(p1[0], p2[0])

		for(let x = Math.floor(sx); x <= fx; x++) {
			const y = -Math.round((line[0]*x + line[2]) / line[1])
			if(y >= 0 && y < img.shape[0]) {
				img.set(0, y, x, 0)
				img.set(0, y, x, 1)
				img.set(0, y, x, 2)
			}
		}
	}
	return img
}

function test(image_num) {
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
