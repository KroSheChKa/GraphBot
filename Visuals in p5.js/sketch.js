var rows = 150;
var cols = 150;
var grid = [];

var openSet = [];
var closedSet = [];

var w,h;
var path = [];
var nosolution = false;

var circles = [];

var fps_counter = 0;

var canvas_width = 750;
var canvas_height = 750;

function distTo(ax, ay, bx, by) {
  return Math.sqrt((ax - bx)*(ax - bx) + (ay - by)*(ay - by));
}

function heuristic(a, b) {
  
  var d = distTo(a.i, a.j, b.i, b.j)
  
  var margin = 5; // how close
  var weight = 1.8 // affraidness
  var penalty = 0;
  
  for (let i = 0; i < circles.length; i++) {
    let cx = circles[i][0];
    let cy = circles[i][1];
    let r = circles[i][2];

    let dToCircle = distTo(a.i, a.j, cx, cy);
    if (dToCircle < r + margin) {
      penalty += (r + margin - dToCircle);
    }
  }

  return d + penalty * weight;
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
  
  this.inOpenSet = false;
  this.inClosedSet = false;

  var if_1 = false;
  
  for (let p = 0; p < circles.length; p++) {
    x = circles[p][0];
    y = circles[p][1];
    r = circles[p][2];
    if_1 = (i - x)*(i - x) + (j - y)*(j - y) < r*r;
    if (if_1) {
      this.wall = true;
      break;
    }
  }
  
  this.show = function(col, layer = undefined) {
    if (layer == bgLayer) {
      bgLayer.stroke(0);
      if (this.wall == true) {
        bgLayer.fill(0);
      } else {
        bgLayer.fill(col);
      }
      bgLayer.rect(this.i*w, this.j*h, w, h);
    } else {
      stroke(0);
      fill(col);
      rect(this.i*w, this.j*h, w, h);
    }
  }
  
  this.addNeighbors = function(grid) {
    // no left movement
    if (i < cols - 1) {
      this.neighbors.push(grid[i+1][j]);
    }
    if (j < rows - 1) {
      this.neighbors.push(grid[i][j+1]);
    }
    if (j > 0) {
      this.neighbors.push(grid[i][j-1]);
    }
    if (j < rows - 1 && i < cols - 1) {
      this.neighbors.push(grid[i+1][j+1]);
    }  
    if (j > 0 && i < cols - 1) {
      this.neighbors.push(grid[i+1][j-1]);
    }
  }
}

function generatCircles(count) {
  for (let i = 0; i < count; i++) {
    var r = random(5, cols/9);
    var x = random(cols*0.1, cols*0.9);
    var y = random(rows*0.1, rows*0.9);
    circles.push([x, y, r])
  }
}

function setup() {
  createCanvas(canvas_width, canvas_height);
  bgLayer = createGraphics(canvas_width, canvas_height);
  
  w = width/cols;
  h = height/rows;
  
  strokeWeight(w / 4);
  bgLayer.background(255)
  bgLayer.noFill();
  bgLayer.strokeWeight(w / 4);
  bgLayer.stroke(255);
  
  generatCircles(25);
  
  for (let i = 0; i < cols; i++) {
    grid[i] = [];
    for (let j = 0; j < rows; j++) {
      grid[i][j] = new Spot(i, j, circles);
      grid[i][j].show(255, bgLayer)
    }
  }
  
  for (let i = 0; i < cols; i++) {
    for (let j = 0; j < rows; j++) {
      grid[i][j].addNeighbors(grid);
    }
  }
  
  start = grid[0][0];
  end = grid[cols-1][rows-1];
  
  end.show(color(255, 215, 0), bgLayer);
  start.show(color(255, 215, 0), bgLayer);
  
  openSet.push(start);
  start.inClosedSet = true;
  
  start.wall = false;
  end.wall = false;
}

function draw() {
  
  if (openSet.length > 0) {
    // great we back on
    var winner = 0;
    
    for (let i = 0; i < openSet.length; i++) {
      if (openSet[i].f < openSet[winner].f) {
        winner = i;
      }
    }

    var current = openSet[winner];
    
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
      
      if (!neighbor.inClosedSet && !neighbor.wall) {
        var tempG = current.g + heuristic(neighbor, current);

        var newPath = false;
        if (neighbor.inOpenSet) {
          if (tempG < neighbor.g) {
            neighbor.g = tempG
            newPath = true;
          }
        } else {
          neighbor.g = tempG;
          newPath = true;
          neighbor.inOpenSet = true;
          openSet.push(neighbor);
        }
        
        if (newPath) {
          neighbor.h = heuristic(neighbor, end);
          neighbor.f = neighbor.g + neighbor.h*neighbor.h;
          neighbor.previous = current;
        }
      }
    }
  } else {
    console.log("No solution!");
    nosolution = true;
    noLoop();
  }
  
  
  // drawing
  if (fps_counter % 5 == 0) {
    image(bgLayer, 0, 0);
    
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
    stroke(255);
    beginShape();
    for (let i = 0; i < path.length; i++) {
      vertex(path[i].i*w + w/2, path[i].j*h + h/2);
    }
    endShape();
    stroke(0);

    }
  
  fps_counter +=1;
  
}
