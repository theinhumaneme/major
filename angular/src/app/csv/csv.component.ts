import { Component, OnInit } from '@angular/core';

interface Model {
  model: string;
  viewValue: string;
}

@Component({
  selector: 'app-csv',
  templateUrl: './csv.component.html',
  styleUrls: ['./csv.component.css']
})

export class CsvComponent implements OnInit {
  
  models: Model[] = [
    {model: 'hyper', viewValue: 'Optimal Model'},
    {model: 'lr', viewValue: 'Linear Regression'},
    {model: 'dtc', viewValue: 'Decision Tree Classifier'},
    {model: 'gbc', viewValue: 'Gradient Boosting Classifier'},
    {model: 'knn', viewValue: 'K Nearest Neighbors'},
    {model: 'rfc', viewValue: 'Random Forest Classifer'},
    {model: 'svc', viewValue: 'Support Vector Machine'},
    

  ];
  constructor() { }

  ngOnInit(): void {
  }

}
