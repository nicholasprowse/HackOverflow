
class Quaternion {

	constructor(w) {
		this.w = w
	}

	conjugate() {
		return new Quaternion([w[0], -w[1], -w[2], -w[3]])
	}

	// u = w*v
	mult(v) {
		const w = this.w
		u = new Array(4).fill(0)
		u[0] = w[0]*v[0] - w[1]*v[1] - w[2]*v[2] - w[3]*v[3]
		u[1] = w[0]*v[1] + w[1]*v[0] + w[2]*v[3] - w[3]*v[2]
		u[2] = w[0]*v[2] + w[2]*v[0] + w[3]*v[1] - w[1]*v[3]
		u[3] = w[0]*v[3] + w[3]*v[0] + w[1]*v[2] - w[2]*v[1]
		return new Quaternion(w)
	}

}
