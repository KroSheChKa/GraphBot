var rows = 100;
var cols = 100;
var grid = [];

var openSet = [];
var closedSet= [];

var w,h;
var path = [];
var nosolution = false;

var circles = [];

function heuristic(a, b) {
  var d = dist(a.i, a.j, b.i, b.j)
  // var d = abs(a.i - b.i) + abs(a.j, b.j)
  return d
}

function removeFromArray(arr, el) {
  for (let i = arr.length - 1; i >= 0; i--) {
    if (arr[i] == el) {
      arr.splice(i, 1);
    }
  }
}

function Spot(i, j, circles) {
  this.i = i;
  this.j = j;
  this.f = 0;
  this.g = 0;
  this.h = 0;
  this.neighbors = [];
  this.previous = undefined;
  this.wall = false;
  this.circles = circles;
  
  // var if_1 = random(1) < 0.5;
  var if_1 = false;
  
  for (let p = 0; p < circles.length; p++) {
    x = circles[p][0];
    y = circles[p][1];
    r = circles[p][2];
    if_1 = pow(i - x, 2) + pow(j - y, 2) < r*r;
    if (if_1 == true) {
      break;
    }
  }
  
  if (if_1 & !(i == 0 & j == 0) & !(i == cols - 1 & j == rows-1)){
    this.wall = true;
  }
  
  this.show = function(col) {
    fill(col);
    if (this.wall == true) {
      fill(0)
    }
    rect(this.i*w, this.j*h, w, h);
  }
  
  this.addNeighbors = function(grid) {
    if (i < cols - 1) {
      this.neighbors.push(grid[i+1][j]);
    }
    // no left movement
    // if (i > 0) {
    //   this.neighbors.push(grid[i-1][j]);
    // }
    if (j < rows - 1) {
      this.neighbors.push(grid[i][j+1]);
    }
    if (j > 0) {
      this.neighbors.push(grid[i][j-1]);
    }
    
    if (j < rows - 1 && i < cols - 1) {
      this.neighbors.push(grid[i+1][j+1]);
    }
    // no left movement
    // if (j > 0 && i > 0) {
    //   this.neighbors.push(grid[i-1][j-1]);
    // }
    
    if (j > 0 && i < cols - 1) {
      this.neighbors.push(grid[i+1][j-1]);
    }
    // no left movement
    // if (j < rows - 1 && i > 0) {
    //   this.neighbors.push(grid[i-1][j+1]);
    // }
  }
}

function generatCircles(count) {
  for (let i = 0; i < count; i++) {
    var r = random(5, cols/9);
    var x = random(cols*0.1, cols*0.9);
    var y = random(rows*0.1, rows*0.9);
    circles.push([x, y, r])
  }
  // return circles;
}

function setup() {
  createCanvas(670, 670);
  
  w = width/cols;
  h = height/rows;
  
  generatCircles(25);
  
  
  for (let i = 0; i < cols; i++) {
    grid[i] = [];
    for (let j = 0; j < rows; j++) {
      grid[i][j] = new Spot(i, j, circles);
    }
  }
  
  for (let i = 0; i < cols; i++) {
    for (let j = 0; j < rows; j++) {
      grid[i][j].addNeighbors(grid);
    }
  }
  
  start = grid[0][0];
  end = grid[cols-1][rows-1];
  
  openSet.push(start);
}

function draw() {
  background(0);
  
  if (openSet.length > 0) {
    // great we back on
    var winner = 0;
    
    for (let i = 0; i < openSet.length; i++) {
      if (openSet[i].f < openSet[winner].f) {
        winner = i;
      }
    }
    

    var current = openSetHeap.extractMin();


    
    //exit
    if (current === end) {
      console.log("Done!");
      noLoop();
    }
    
    removeFromArray(openSet, current);
    closedSet.push(current);
    
    var neighbors = current.neighbors;
    for(var i = 0; i < neighbors.length; i++) {
      var neighbor = neighbors[i];
      
      if (!closedSet.includes(neighbor) && !neighbor.wall) {
        var tempG = current.g + heuristic(neighbor, current);

        var newPath = false;
        if (openSet.includes(neighbor)) {
          if (tempG < neighbor.g) {
            neighbor.g = tempG
            newPath = true;
          }
        } else {
          neighbor.g = tempG;
          newPath = true;
          openSet.push(neighbor);
        }
        
        if (newPath) {
          neighbor.h = pow(heuristic(neighbor, end), 11);
          neighbor.f = neighbor.g + neighbor.h;
          neighbor.previous = current;
        }
      }
    }
  } else {
    console.log("No solution!");
    nosolution = true;
    noLoop();
  }
  
  
  for (let i = 0; i < cols; i++) {
    for (let j = 0; j < rows; j++) {
      grid[i][j].show(255);
    }
  }
  
  for (let i = 0; i < closedSet.length; i++) {
    closedSet[i].show(color(230, 0, 0))
  }
      
  for (let i = 0; i < openSet.length; i++) {
    openSet[i].show(color(0, 230, 0))
  }
  
  if (!nosolution) {
    path = [];
    var temp = current;
    while (temp.previous) {
    path.push(temp.previous);
    temp = temp.previous;
    }
  }
  

  for (let i = 0; i < path.length; i++) {
    path[i].show(color(0, 0, 230));
  }
  
  noFill();
  strokeWeight(w / 4);
  stroke(255);
  beginShape();
  for (let i = 0; i < path.length; i++) {
    vertex(path[i].i*w + w/2, path[i].j*h + h/2);
  }
  endShape();
  stroke(0);
  
  end.show(color(255, 215, 0));
  start.show(color(255, 215, 0));
}