// This just represents 3 faces of a cube, as a photo of a cube can only ever see 3 faces
class Cube {

	static points = [
		[0, 0, 0],
		[1, 0, 0],
		[0, 1, 0],
		[0, 0, 1],
		[1, 1, 0],
		[1, 0, 1],
		[0, 1, 1],
		[0, 0, 1/3],
		[0, 1, 1/3],
		[1, 0, 1/3],
		[0, 0, 2/3],
		[0, 1, 2/3],
		[1, 0, 2/3],
		[0, 1/3, 0],
		[1, 1/3, 0],
		[0, 1/3, 1],
		[0, 2/3, 0],
		[1, 2/3, 0],
		[0, 2/3, 1],
		[1/3, 0, 0],
		[1/3, 1, 0],
		[1/3, 0, 1],
		[2/3, 0, 0],
		[2/3, 1, 0],
		[2/3, 0, 1]
	]

	static lines = [
		[0, 1],
		[0, 2],
		[0, 3],
		[1, 4],
		[1, 5],
		[2, 4],
		[2, 6],
		[3, 5],
		[3, 6],
		[7, 8],
		[7, 9],
		[10, 11],
		[10, 12],
		[13, 14],
		[13, 15],
		[16, 17],
		[16, 18],
		[19, 20],
		[19, 21],
		[22, 23],
		[22, 24]
	]

	constructor(pos, angle) {
		this.pos = pos
		this.angle = angle
	}

	transformPoint(p) {
		// Shorthand for cos and sin of each angle
		let [a, b, c] = this.angle
		let [ca, cb, cc] = [Math.cos(a), Math.cos(b), Math.cos(c)]
		let [sa, sb, sc] = [Math.sin(a), Math.sin(b), Math.sin(c)]
		// Roll Pitch Yaw matrix
		const e = [
			[ca * cb, -cb * sa, sb],
			[cc * sa + ca * sb * sc, ca * cc - sa * sb * sc, -cb * sc],
			[-ca * cc * sb + sa * sc, cc * sa * sb + ca * sc, cb * cc]
		]

		const transform = [0, 0, 0]
		for (let i = 0; i < 3; i++) {
			for (let j = 0; j < 3; j++)
				transform[i] += e[i][j] * p[j]
			transform[i] += this.pos[i]
		}

		return transform
	}

	static projectPoint(p, f, u) {
		return [f * p[0] / p[2] + u[0], f * p[1] / p[2] + u[1]]
	}

	getTotalError(lines, f, u) {
		const p = new Array(Cube.points.length)
		for (let i = 0; i < Cube.points.length; i++)
			p[i] = Cube.projectPoint(
				this.transformPoint(
					Cube.points[i]), f, u)

		// TODO: Find some way to optimise the allocation of line segments to lines
		// One idea (not necessarily optimal, but should be close) is to compute all
		// 21*21 line correspondances in a 2D array, assign lines using a greedy
		// algorithm (same as current), then iterate through every pair of lines.
		// If swapping them is an improvement, then swap them, otherwise don't.
		// Continue iterating through pairs, until an entire iteration occurs without change

		// Compute correspondances between every line pair
		let n = Cube.lines.length
		const correspondances = new Array(n * n)

		for(let i = 0; i < n; i++) {
			let [u, v] = [p[Cube.lines[i][0]], p[Cube.lines[i][1]]]
			for(let j = 0; j < n; j++) {
				correspondances[n*i + j] = lineCorrespondence(u, v, lines[j])
			}
		}

		// Sort the correspondances
		let index = [...Array(n*n).keys()]
		index.sort((a, b) => correspondances[a] - correspondances[b])

		// Initialise line pairs to a good initial pairing by progressivly
		// selecting the minimum correspondance from two lines yet to be selected
		const linePairs = new Array(n)
		const found1 = new Array(n).fill(false)
		const found2 = new Array(n).fill(false)
		let idx = 0
		for(let i = 0; i < n; i++) {
			while(true) {
				const c = correspondances[index[idx]]
				const [i1, i2] = [Math.floor(index[idx] / n), index[idx] % n]
				idx++
				if(!found1[i1] && !found2[i2]) {
					linePairs[i] = [i1, i2]
					found1[i1] = true
					found2[i2] = true
					break
				}
			}
		}

		// repeatedly swap elements until no improvements can be made
		let swapped = true
		while(swapped) {
			swapped = false
			for(let i = 0; i < n; i++) {
				for(let j = 0; j < n; j++) {
					let [lp1, lp2] = [linePairs[i], linePairs[j]]
					const before = correspondances[lp1[0]*n + lp1[1]] + correspondances[lp2[0]*n + lp2[1]]
					const after = correspondances[lp2[0]*n + lp1[1]] + correspondances[lp1[0]*n + lp2[1]]
					if(after < before) {
						const temp = lp1[0]
						lp1[0] = lp2[0]
						lp2[0] = temp
						swapped = true
					}
				}
			}
		}

		let error = 0
		for(let lp of linePairs)
			error += correspondances[lp[0] * n + lp[1]]
		return error
	}

	drawOnImage(img, f, c) {
		const projectedPoints = new Array(Cube.points.length)
		for (let i = 0; i < Cube.points.length; i++)
			projectedPoints[i] = Cube.projectPoint(
				this.transformPoint(
					Cube.points[i]), f, [img.shape[1] / 2, img.shape[0] / 2])

		for (let line of Cube.lines)
			img = drawLineSegment(img, projectedPoints[line[0]], projectedPoints[line[1]], c)

		return img
	}

}

// Sorts x according to the values in f
function sortAccordingTo(x, f) {
	let index = [...Array(f.length).keys()]
	index.sort((a, b) => f[a] - f[b])
	const out = new Array(x.length)
	for(let i = 0; i < x.length; i++)
		out[i] = x[index[i]]
	return out
}

// Returns a value indicating how closely a line segment coincides with a line
// Line segment defined by 2 points, u and v, and line defined by l
function lineCorrespondence(u, v, l) {
	u = [u[0], u[1], 1]
	v = [v[0], v[1], 1]
	const d = sub(u, v)
	const num = Math.sqrt(dot(d, d)) * (dot(l, v)**3 - dot(l, u)**3)
	const denom = dot([-l[0], l[1], -l[2]], d)**2 * dot(l, sub(v, u))
	return num / denom
}

function sub(a, b) {
	let diff = new Array(a.length);
	for(let i = 0; i < a.length; i++)
		diff[i] = a[i] - b[i];
	return diff;
}

function dot(a, b) {
	let sum = 0;
	for(let i = 0; i < a.length; i++)
		sum += a[i] * b[i];
	return sum;
}

for(let c of Cube.points) {
	c[0] -= 0.5
	c[1] -= 0.5
	c[2] -= 0.5
}
