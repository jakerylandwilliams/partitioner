var margin_lines = {top: 70, right: 20, bottom: 40, left: 50},
    width_lines = 370 - margin_lines.left - margin_lines.right,
    height_lines = 410 - margin_lines.top - margin_lines.bottom;

var x_lines = d3.scale.linear();

var y_lines = d3.scale.linear();

var xAxis_lines = d3.svg.axis()
    .scale(x_lines)
    .orient("bottom");

var yAxis_lines = d3.svg.axis()
    .scale(y_lines)
    .orient("left");

var line = d3.svg.line()
    .x(function(d) { return x_lines(d.rank); })
    .y(function(d) { return y_lines(d.freq); });

var svg_lines = d3.select("#lines").append("svg")
    .attr("width", width_lines + margin_lines.left + margin_lines.right)
    .attr("height", height_lines + margin_lines.top + margin_lines.bottom)
    .append("g")
    .attr("transform", "translate(" + margin_lines.left + "," + margin_lines.top + ")");

function make_lines(ID){

    xmax = lines[lines.length-1]["rank"];
    ymax = lines[0]["freq"];

    x_lines.domain([-xmax*0.01,xmax]).range([0, width_lines]);
    y_lines.domain([-ymax*0.01,ymax]).range([height_lines,0]);

    svg_lines.append("g")
	.attr("class", "x axis")
	.attr("transform", "translate(0," + height_lines + ")")
	.call(xAxis_lines)

    svg_lines.append("text")
	.attr("class", "label")
	.attr("x", width_lines)
	.attr("y", height_lines + margin_lines.bottom - 10)
	.style("text-anchor", "end")
	.text("Log₁₀(r)");

    svg_lines.append("g")
	.attr("class", "y axis")
	.call(yAxis_lines)

    svg_lines.append("text")
	.attr("class", "label")
	.attr("transform", "rotate(-90)")
	.attr("y", -50)
	.attr("dy", "1.5em")
	.style("text-anchor", "end")
	.text("Log₁₀(f)");

    svg_lines.append("path")
        .attr("d", line(model))
        .style("stroke-width", 2)
        .style("stroke", "rgb(200,0,0)")
        .style("fill", "none")
	.style("stroke-dasharray", ("3, 3"));    
    
    svg_lines.append("path")
        .attr("d", line(lines))
        .style("stroke-width", 2)
        .style("stroke", "rgb(0,0,0)")
        .style("fill", "none")

    svg_lines.append("text")
	.attr("class","title")
        .attr("x", (width_lines / 2))
        .attr("y", -30 )
        .attr("text-anchor", "middle")
        .style("font-size", "14px")
        .text(data[ID]['name']);

    svg_lines.append("text")
	.attr("class", "label")
	.attr("x", width_lines/1.25)
	.attr("y", height_lines/6)
	.style("text-anchor", "end")
	.text("R²: "+rsq.toFixed(2));    
}

