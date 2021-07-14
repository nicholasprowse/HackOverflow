const fs = require('fs');
const url = require('url');
const util = require('util');
const http = require('http');
let {PythonShell} = require('python-shell');

const server = http.createServer((req,res) => {
	if (req.url === '/') {
		console.log(req.url);
		res.writeHead(200, {'Content-Type' : 'text/html'});
		var myReadStream = fs.createReadStream('index.html','utf8');
		myReadStream.pipe(res)
	}

	if (req.url === '/photo'){
		console.log(req.url);
		res.writeHead(200, {'Content-Type' : 'text/html'});
		var myReadStream = fs.createReadStream('photo.html','utf8');
		myReadStream.pipe(res)
	}

	if (req.url === '/solve'){
		console.log(req.url);
		res.writeHead(200, {'Content-Type' : 'text/html'});
		var myReadStream = fs.createReadStream('solve.html','utf8');
		myReadStream.pipe(res)
	}

	if (req.url === '/calc'){
		

		// Convert rubix cube in the following format
		let sampleInput = "OOWOOWOOWYYYWWRGGGOBBYYYWWRGGGOBBYYYWWRGGGOBBRRBRRBRRB" 


		//path depnds on host machine that we will run for the demo or however it will be deployed
		let options = {
			scriptPath: './',
			args: [sampleInput],
			pythonPath: '/opt/homebrew/bin/python3'
		};
		var test = new PythonShell('rubixSolver.py',options);
		test.on('message', function(message) {
			res.setHeader('Access-Control-Allow-Origin', '*');
			res.writeHead(200, "OK", {"Content-Type":"text/plain"});
			res.write(message)
			console.log(message)
			res.end()
		});
		
	}

});

server.listen(1000);
console.log('Listening on port 1000');
