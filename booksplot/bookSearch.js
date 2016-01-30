var data = {};
var points = [];
var lines = [];
var model = [];
var rsq = 0;
d3.json("booksdata.json", function(booksdata) {

    // generate a list for auto complete
    allbooks = [];
    for (ID in booksdata){
	data[ID] = booksdata[ID];
	allbooks.push({label: booksdata[ID]['name'], value: ID});
    };

    // Generate the points for the scatter plot
    for (var key in data){
    	//if (data[key]["qInf-rsq"] > 0.4 & data[key]["1.0-rsq"] > 0.4){
    	points.push([x_scatter(+data[key]["qInf-rsq"]),y_scatter(+data[key]["1.0-rsq"])]);
    	//}
    };

    // initialize both plots with Moby Dick
    ID = "englishGutenbergBooks/2701.txt";

    // Generate the lines for the rank-freq plot
    if (data[ID]["qInf-rsq"] >= data[ID]["1.0-rsq"]){
	q = "qInf";
    }
    else{
	q = "1.0";
    }
    rsq = +data[ID][q+"-rsq"];

    N = +data[ID][q+"-plotNumbers"][data[ID][q+"-plotSizes"].length-1];
    sizes = [];
    numbers = [];
    prevsize = +data[ID][q+"-plotSizes"][0];
    prevnumber = 0;    
    
    for (i = 0; i < data[ID][q+"-plotSizes"].length; i++) {
	cumnumber = +data[ID][q+"-plotNumbers"][i];
	size = +data[ID][q+"-plotSizes"][i];
	number =  cumnumber - prevnumber
	if (prevsize != size){
	    sizes.push(size);
	    numbers.push(number);
	}
	else{
	    sizes[sizes.length-1] = size;
	    numbers[numbers.length-1] = number + 1;
	}
	lines.push({'rank': Math.log10(cumnumber),'freq': Math.log10(size)});
	prevsize = size;
	prevnumber = cumnumber;
    }
    M = 0;
    for (i = 0; i < sizes.length; i++) {
	M += sizes[i]*numbers[i];
    }
    if (M){
	m = 1 - (N/M);
	b = m*Math.log10(N);
    }
    else{
	m = 0;
	b = 0;
    }
    model = [{'rank': 0,'freq': -m*0+b},{'rank': Math.log10(N),'freq': -m*Math.log10(N)+b}];

    // make plots
    make_scatter(ID);
    make_lines(ID);

    // use Jquery autocomplete
    ////////////////////////////////
    $( "#book_search_box" ).autocomplete({
	source: allbooks,
	minLength: 3
    });

    // submit business button
    $("#book_search_box").keyup(function (e) {
	if (e.keyCode == 13) {
	    plotbook();
	}
    });
});

function plotbook(){
    ID = $('#book_search_box').val();
    
    lines = [];
    if (data[ID]["qInf-rsq"] >= data[ID]["1.0-rsq"]){
	q = "qInf";
    }
    else{
	q = "1.0";
    }
    rsq = +data[ID][q+"-rsq"];

    N = +data[ID][q+"-plotNumbers"][data[ID][q+"-plotSizes"].length-1];
    sizes = [];
    numbers = [];
    prevsize = +data[ID][q+"-plotSizes"][0];
    prevnumber = 0;    
    
    for (i = 0; i < data[ID][q+"-plotSizes"].length; i++) {
	cumnumber = +data[ID][q+"-plotNumbers"][i];
	size = +data[ID][q+"-plotSizes"][i];
	number =  cumnumber - prevnumber
	if (prevsize != size){
	    sizes.push(size);
	    numbers.push(number);
	}
	else{
	    sizes[sizes.length-1] = size;
	    numbers[numbers.length-1] = number + 1;
	}
	lines.push({'rank': Math.log10(cumnumber),'freq': Math.log10(size)});
	prevsize = size;
	prevnumber = cumnumber;
    }
    M = 0;
    for (i = 0; i < sizes.length; i++) {
	M += sizes[i]*numbers[i];
    }
    if (M){
	m = 1 - (N/M);
	b = m*Math.log10(N);
    }
    else{
	m = 0;
	b = 0;
    }
    model = [{'rank': 0,'freq': -m*0+b},{'rank': Math.log10(N),'freq': -m*Math.log10(N)+b}];
    
    svg_scatter.selectAll("*").remove();
    svg_lines.selectAll("*").remove();    
    make_scatter(ID);
    make_lines(ID);
}
