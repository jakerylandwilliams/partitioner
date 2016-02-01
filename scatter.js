var margin_scatter = {top: 70, right: 20, bottom: 40, left: 50},
    w_scatter = 370 - margin_scatter.left - margin_scatter.right,
    h_scatter = 410 - margin_scatter.top - margin_scatter.bottom;

// var margin_scatter = {top: 105, right: 30, bottom: 60, left: 75},
//     w_scatter = 370 + 185 - margin_scatter.left - margin_scatter.right,
//     h_scatter = 410 + 205 - margin_scatter.top - margin_scatter.bottom;

var color_scatter = d3.scale.linear()
    .domain([0, 64])
    .range(["lightgray", "black"])
    .interpolate(d3.interpolateLab);

var x_scatter = d3.scale.linear()
    .domain([-0.02, 1])
    .range([0, w_scatter]);

var y_scatter = d3.scale.linear()
    .domain([-0.02, 1])
    .range([h_scatter, 0]);

var xAxis_scatter = d3.svg.axis()
    .scale(x_scatter)
    .orient("bottom");

var yAxis_scatter = d3.svg.axis()
    .scale(y_scatter)
    .orient("left");

var line_scatter = d3.svg.line()
    .x(function(d) { return x_scatter(d.x); })
    .y(function(d) { return y_scatter(d.y); });    

var hexbin = d3.hexbin()
    .size([w_scatter, h_scatter])
    .radius(2);

var svg_scatter = d3.select("#scatter").append("svg")
    .attr("width", w_scatter + margin_scatter.left + margin_scatter.right)
    .attr("height", h_scatter + margin_scatter.top + margin_scatter.bottom)
    .append("g")
    .attr("transform", "translate(" +margin_scatter.left+ "," +margin_scatter.top+ ")");

function make_scatter(ID){

    svg_scatter.append("g")
	.attr("class", "x axis")
	.attr("transform", "translate(0," + h_scatter + ")")
	.call(xAxis_scatter)

    svg_scatter.append("text")
	.attr("class", "label")
	.attr("x", w_scatter)
	.attr("y", h_scatter + margin_scatter.bottom - 10)
	.style("text-anchor", "end")
	.text("Phrase R²");

    svg_scatter.append("g")
	.attr("class", "y axis")
	.call(yAxis_scatter)

    svg_scatter.append("text")
	.attr("class", "label")
	.attr("transform", "rotate(-90)")
	.attr("y", -50)
	.attr("dy", "1.5em")
	.style("text-anchor", "end")
	.text("Word R²");

    svg_scatter.selectAll(".hex")
	.data(data)
	.enter().append("clipPath")
	.attr("id", "clip")
	.append("rect")
	.attr("class", "mesh")
	.attr("width", w_scatter)
	.attr("height", h_scatter);

    svg_scatter.append("g")
	.attr("clip-path", "url(#clip)")
	.selectAll(".hexagon")
	.data(hexbin(points))
	.enter().append("path")
	.attr("class", "hexagon")
	.attr("d", hexbin.hexagon())
	.attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; })
	.style("fill", function(d) { return color_scatter(d.length); });

    // Plot the book of interest
    pdata = [{'x': +data[ID]["qInf-rsq"],'y': +data[ID]["1.0-rsq"]}];
    svg_scatter.selectAll("dot")
        .data(pdata)
        .enter().append("circle")
        .attr("r", 3)
        .attr("cx", function(d) { return x_scatter(d.x); })
        .attr("cy", function(d) { return y_scatter(d.y); })
	.attr("fill", "rgb(0,0,200)");

    // plot the line y = x
    identity = [{'x': 0, 'y': 0},{'x': 1, 'y': 1}];
    svg_scatter.append("path")
        .attr("d", line_scatter(identity))
        .style("stroke-width", 2)
        .style("stroke", "rgb(200,0,0)")
        .style("fill", "none")
	.style("stroke-dasharray", ("3, 3"));
    
}

var title_scatter = d3.select("svg").append("g")
    .attr("transform", "translate(" +margin_scatter.left+ "," +margin_scatter.top+ ")")
    .attr("class","title");

title_scatter.append("text")
    .attr("class","title")
    .attr("x", (w_scatter / 2))
    .attr("y", -30 )
    .attr("text-anchor", "middle")
    .style("font-size", "14px")
    .text("Goodness of fit");
