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

	constructor(pos, angle, f, u0) {
		this.pos = pos
		this.angle = angle
		this.f = f
		this.u0 = u0
		this.rpy = []
		this.dirty = true
	}

	setAngle(angle) {
		this.angle = angle
		this.dirty = true
	}

	createRPYMatrix() {
		if(!this.dirty)
			return

		// Shorthand for cos and sin of each angle
		let [a, b, c] = this.angle
		let [ca, cb, cc] = [Math.cos(a), Math.cos(b), Math.cos(c)]
		let [sa, sb, sc] = [Math.sin(a), Math.sin(b), Math.sin(c)]
		// Roll Pitch Yaw matrix
		this.rpy = [
			[ca * cb, -cb * sa, sb],
			[cc * sa + ca * sb * sc, ca * cc - sa * sb * sc, -cb * sc],
			[-ca * cc * sb + sa * sc, cc * sa * sb + ca * sc, cb * cc]
		]

		this.rpyDeriv = [[
			[-sa * cb, -cb * ca, 0],
			[cc * ca - sa * sb * sc, -sa * cc - ca * sb * sc, 0],
			[sa * cc * sb + ca * sc, cc * ca * sb - sa * sc, 0]
		],
		[
			[-ca * sb, sb * sa, cb],
			[cc * sa + ca * cb * sc, ca * cc - sa * cb * sc, sb * sc],
			[-ca * cc * cb + sa * sc, cc * sa * cb + ca * sc, -sb * cc]
		],
		[
			[0, 0, 0],
			[-sc * sa + ca * sb * cc, -ca * sc - sa * sb * cc, -cb * cc],
			[ca * sc * sb + sa * cc, -sc * sa * sb + ca * cc, -cb * sc]
		]]
		this.dirty = false
	}

	transformPoint(p) {
		this.createRPYMatrix()

		const transform = [0, 0, 0]
		for (let i = 0; i < 3; i++) {
			for (let j = 0; j < 3; j++)
				transform[i] += this.rpy[i][j] * p[j]
			transform[i] += this.pos[i]
		}

		return transform
	}

	projectPoint(p) {
		return [this.f * p[0] / p[2] + this.u0[0], this.f * p[1] / p[2] + this.u0[1], 1]
	}

	getTotalError(lines) {
		const n = Cube.lines.length
		const m = lines.length
		const correspondances = this.getLineCorrespondences(lines)
		const linePairs = this.assignLines2(correspondances)

		let error = 0
		for(let lp of linePairs)
			error += correspondances[lp[0] * m + lp[1]]
		return error
	}

	getErrorDerivativeWRTPosition(lines) {
		const correspondances = this.getLineCorrespondences(lines)
		const linePairs = this.assignLines2(correspondances)
		let deriv = [0, 0, 0]
		for(let lp of linePairs)
			deriv = add(deriv, this.de_dx(lines[lp[1]], lp[0]))
		return deriv
	}

	getErrorDerivativeWRTAngle(lines) {
		const correspondances = this.getLineCorrespondences(lines)
		const linePairs = this.assignLines2(correspondances)
		let deriv = [0, 0, 0]
		for(let lp of linePairs)
			deriv = add(deriv, this.de_da(lines[lp[1]], lp[0]))
		return deriv
	}

	getErrorDerivativeWRTFocalLength(lines) {
		const correspondances = this.getLineCorrespondences(lines)
		const linePairs = this.assignLines2(correspondances)
		let deriv = 0
		for(let lp of linePairs)
			deriv += this.de_df(lines[lp[1]], lp[0])
		return deriv
	}

	stepParameters(lines, lr) {
		const pos = this.getErrorDerivativeWRTPosition(lines)
		const angle = this.getErrorDerivativeWRTAngle(lines)
		const f = this.getErrorDerivativeWRTFocalLength(lines)
		this.pos[0] -= lr * pos[0]
		this.pos[1] -= lr * pos[1]
		this.pos[2] -= lr * pos[2]
		this.angle[0] -= lr * angle[0]
		this.angle[1] -= lr * angle[1]
		this.angle[2] -= lr * angle[2]
		this.f -= lr * f
		this.dirty = true
	}

	// Derivative of the error function with respect to the position x = (x, y, z)
	// for a given line l, and line segment s where f is the focal length
	de_dx(l, s) {
		const p1 = this.transformPoint(Cube.points[Cube.lines[s][0]])
		const p2 = this.transformPoint(Cube.points[Cube.lines[s][1]])
		const u = this.projectPoint(p1)
		const v = this.projectPoint(p2)
		const dedu = de_du(l, u, v)
		const dedv = de_du(l, v, u)
		const d = new Array(3)
		d[0] = this.f*(dedu[0]/p1[2] + dedv[0]/p2[2])
		d[1] = this.f*(dedu[1]/p1[2] + dedv[1]/p2[2])
		d[2] = -this.f*((p1[0]*dedu[0] + p1[1]*dedu[1])/p1[2]**2 + (p2[0]*dedv[0] + p2[1]*dedv[1])/p2[2]**2)
		return d
	}

	de_da(l, s) {
		const f = this.f
		const x = Cube.points[Cube.lines[s][0]]
		const y = Cube.points[Cube.lines[s][1]]
		const p1 = this.transformPoint(x)
		const p2 = this.transformPoint(y)
		const u = this.projectPoint(p1)
		const v = this.projectPoint(p2)
		const dedu = de_du(l, u, v)
		const dedv = de_du(l, v, u)

		const dp1_da = [new Array(3).fill(0), new Array(3).fill(0), new Array(3).fill(0)]
		const dp2_da = [new Array(3).fill(0), new Array(3).fill(0), new Array(3).fill(0)]
		for(let aInd = 0; aInd < 3; aInd++)
			for(let row = 0; row < 3; row++)
				for(let i = 0; i < 3; i++) {
					dp1_da[row][aInd] += this.rpyDeriv[aInd][row][i] * x[i]
					dp2_da[row][aInd] += this.rpyDeriv[aInd][row][i] * y[i]
				}

		const du_dp1 = [
			[f/p1[2], 0, 0],
			[0, f/p1[2], 0],
			[-f*p1[0]/p1[2]**2, -f*p1[1]/p1[2]**2, 0]
		]
		const dv_dp2 = [
			[f/p2[2], 0, 0],
			[0, f/p2[2], 0],
			[-f*p2[0]/p2[2]**2, -f*p2[1]/p2[2]**2, 0]
		]

		const du_da = [new Array(3).fill(0), new Array(3).fill(0), new Array(3).fill(0)]
		const dv_da = [new Array(3).fill(0), new Array(3).fill(0), new Array(3).fill(0)]
		for(let row = 0; row < 3; row++)
			for(let col = 0; col < 3; col++)
				for(let i = 0; i < 3; i++) {
					du_da[row][col] += du_dp1[row][i] * dp1_da[i][col]
					dv_da[row][col] += dv_dp2[row][i] * dp2_da[i][col]
				}

		const d = new Array(3).fill(0)
		for(let col = 0; col < 3; col++)
			for(let i = 0; i < 3; i++) {
				d[col] += du_da[i][col] * dedu[i]
				d[col] += dv_da[i][col] * dedv[i]
			}

		return d
	}

	de_df(l, s) {
		const p1 = this.transformPoint(Cube.points[Cube.lines[s][0]])
		const p2 = this.transformPoint(Cube.points[Cube.lines[s][1]])
		const u = this.projectPoint(p1)
		const v = this.projectPoint(p2)
		const dedu = de_du(l, u, v)
		const dedv = de_du(l, v, u)
		return dedu[0]*(p1[0]/p1[2]) + dedu[1]*(p1[1]/p1[2]) + dedv[0]*(p2[0]/p2[2]) + dedv[1]*(p2[1]/p2[2])
	}

	getLineCorrespondences(lines) {
		const p = new Array(Cube.points.length)
		for (let i = 0; i < Cube.points.length; i++)
			p[i] = this.projectPoint(
				this.transformPoint(
					Cube.points[i]))

		// Compute correspondances between every line pair
		const n = Cube.lines.length
		const m = lines.length
		const correspondances = new Array(m * n)

		for(let i = 0; i < n; i++) {
			let [u, v] = [p[Cube.lines[i][0]], p[Cube.lines[i][1]]]
			for(let j = 0; j < m; j++)
				correspondances[m*i + j] = lineCorrespondence(lines[j], u, v)
		}
		return correspondances
	}

	assignLines(correspondances) {
		// Sort the correspondances
		const n = Cube.lines.length
		const m = correspondances.length / n
		let index = [...Array(m*n).keys()]
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

		return linePairs
	}

	assignLines2(correspondances) {
		const n = Cube.lines.length
		const m = correspondances.length / n
		const linePairs = new Array(m)
		for(let j = 0; j < m; j++) {
			let minI = 0
			for(let i = 1; i < n; i++)
				if(correspondances[i * m + j] < correspondances[minI * m + j])
					minI = i

			linePairs[j] = [minI, j]
		}
		return linePairs
	}

	drawOnImage(img, c) {
		const projectedPoints = new Array(Cube.points.length)
		for (let i = 0; i < Cube.points.length; i++)
			projectedPoints[i] = this.projectPoint(
				this.transformPoint(
					Cube.points[i]))

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
function lineCorrespondence(l, u, v) {
	const d = sub(u, v)
	const num = norm(d) * (dot(l, u)**3 - dot(l, v)**3)
	const denom = dot(mult(l, [1, -1, 1]), d)**2 * dot(l, d)
	return num / denom
}

// Computes the derivative of the error with repect to u. For the derivative
// with respect to v, swap the u and v arguments as the error is symmetric
// around u and v
function de_du(l, u, v) {
	const d = sub(u, v)
	const norm_d = norm(d)
	const cube_diff = (dot(l, u)**3 - dot(l, v)**3)
	const neg_l = mult(l, [1, -1, 1])
	const neg_l_dot_d = dot(neg_l, d)


	const num = norm_d * cube_diff
	const denom = neg_l_dot_d**2 * dot(l, d)

	const derivNum = add(mult(d, cube_diff / norm_d), mult(l, 3*norm_d*dot(l, u)**2))
	const derivDenom = add(mult(neg_l, 2*neg_l_dot_d*dot(l, d)), mult(l, neg_l_dot_d**2))

	return mult(sub(mult(derivNum, denom), mult(derivDenom, num)), 1 / denom**2)
}

for(let c of Cube.points) {
	c[0] -= 0.5
	c[1] -= 0.5
	c[2] -= 0.5
}
