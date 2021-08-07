function sub(a, b) {
	let diff = new Array(a.length)
	for(let i = 0; i < a.length; i++)
		diff[i] = a[i] - b[i]
	return diff
}

function add(a, b) {
	let diff = new Array(a.length);
	for(let i = 0; i < a.length; i++)
		diff[i] = a[i] + b[i]
	return diff
}

function normSqr(a) {
	return dot(a, a)
}

function norm(a) {
	return Math.sqrt(normSqr(a))
}

function mult(a, k) {
	let arr = new Array(a.length)
	if(typeof(k) === "number") {
		for(let i = 0; i < a.length; i++)
			arr[i] = a[i] * k
	} else {
		for(let i = 0; i < a.length; i++)
			arr[i] = a[i] * k[i]
	}
	return arr
}

function dot(a, b) {
	let sum = 0
	for(let i = 0; i < a.length; i++)
		sum += a[i] * b[i]
	return sum
}
