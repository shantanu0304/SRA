(function ($)
{
 "use strict";
 var x = $.get('/data');

 var a = x.done(function (params)
 {
   /*----------------------------------------*/
	/*  1.  Bar Chart
	/*----------------------------------------*/

	var ctx = document.getElementById("barchart1");
	var barchart1 = new Chart(ctx, {
		type: 'bar',
		data: {
			labels: params.algokeys,
			datasets: [{
				label: 'Accuracy Comparision',
				data: params.algovalues,
				backgroundColor: [
					'rgba(255, 99, 132, 0.2)',
					'rgb(50,205,50, 0.2)',
					'rgba(255, 206, 86, 0.2)',
					'rgba(75, 192, 192, 0.2)',
					'rgba(255, 99, 132, 0.2)',
					'rgb(50,205,50, 0.2)',
					'rgba(255, 206, 86, 0.2)',
					'rgba(75, 192, 192, 0.2)'
				],
				borderColor: [
					'rgba(255,99,132,1)',
					'rgba(54, 162, 235, 1)',
					'rgba(255, 206, 86, 1)',
					'rgba(75, 192, 192, 1)',
          'rgba(255,99,132,1)',
					'rgba(54, 162, 235, 1)',
					'rgba(255, 206, 86, 1)',
					'rgba(75, 192, 192, 1)'
				],
				borderWidth: 1
			}]
		},
		options: {
			scales: {yAxes: [{ticks: {beginAtZero:true}}]}
		}
	});
   /*----------------------------------------*/
   /*  2.  Bar Chart vertical
   /*----------------------------------------*/
   var ctx = document.getElementById("barchart2");
   console.log("Json Data",params.algokeys);

   var barchart2 = new Chart(ctx, {
 		type: 'line',
 		data: {
 			labels: params["targetrange"],
 			datasets: [{
 				label: "Dataset 1",
 				fill: false,
                 backgroundColor: '#00c292',
 				borderColor: '#00c292',
 				data: params["targetvalue"]
             }]
 		},
 		options: {
 			responsive: true,
 			title:{
 				display:true,
 				text:'Target Variable Distribution'
 			},
 			tooltips: {
 				mode: 'index',
 				intersect: false,
 			},
 			hover: {
 				mode: 'nearest',
 				intersect: true
 			},
 			scales: {
 				xAxes: [{
 					display: true,
 					scaleLabel: {
 						display: true,
 						labelString: params["targetname"]
 					}
 				}],
 				yAxes: [{
 					display: true,
 					scaleLabel: {
 						display: true,
 						labelString: 'Value'
 					}
 				}]
 			}
 		}
 	});



 });
})(jQuery);
