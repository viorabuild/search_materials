/**
 * ESTIMATE CONSTRUCTOR SCRIPT
 * 
 * usage:
 * 1. Ensure you have a sheet named 'DB_Works' with columns: Category, Name, Unit, Price, Description
 * 2. Run the 'setupEstimateConstructor' function.
 */

// Enhanced version with template support and summary
function setupEstimateConstructor() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var dbSheet = ss.getSheetByName("DB_Works");
  
  if (!dbSheet) {
    SpreadsheetApp.getUi().alert("Error: Could not find sheet 'DB_Works'. Please import the CSV first.");
    return;
  }

  // 1. Create or Get Calculator Sheet
  var calcSheet = ss.getSheetByName("Estimate_Calculator");
  if (!calcSheet) {
    calcSheet = ss.insertSheet("Estimate_Calculator");
  } else {
    calcSheet.clear(); // Reset if exists
  }

  // 2. Set Up Headers
  var headers = [["Category", "Work Item", "Description", "Unit", "Price (Est.)", "Quantity", "Total", "Notes"]];
  calcSheet.getRange("A1:H1").setValues(headers).setFontWeight("bold").setBackground("#f3f3f3");
  
  // Freeze header
  calcSheet.setFrozenRows(1);

  // 3. Data Validation (Dropdown for Work Items)
  var lastDbRow = dbSheet.getLastRow();
  if (lastDbRow > 1) {
    var workNamesRange = dbSheet.getRange(2, 2, lastDbRow - 1, 1);
    var rule = SpreadsheetApp.newDataValidation()
      .requireValueInRange(workNamesRange)
      .setAllowInvalid(true) // Allow custom entries
      .setHelpText("Select a valid work item or enter custom text.")
      .build();
      
    // Apply dropdown to Column B (Work Item) for 100 rows
    calcSheet.getRange("B2:B100").setDataValidation(rule);
  }

  // 4. Add Formulas
  for (var i = 2; i <= 100; i++) {
    // A: Category
    calcSheet.getRange("A" + i).setFormula(
      `=IFERROR(INDEX(DB_Works!A:A, MATCH(B${i}, DB_Works!B:B, 0)), "")`
    );
    
    // C: Description
    calcSheet.getRange("C" + i).setFormula(
      `=IFERROR(INDEX(DB_Works!E:E, MATCH(B${i}, DB_Works!B:B, 0)), "")`
    );

    // D: Unit
    calcSheet.getRange("D" + i).setFormula(
      `=IFERROR(INDEX(DB_Works!C:C, MATCH(B${i}, DB_Works!B:B, 0)), "")`
    );

    // E: Price
    calcSheet.getRange("E" + i).setFormula(
      `=IFERROR(INDEX(DB_Works!D:D, MATCH(B${i}, DB_Works!B:B, 0)), "")`
    );

    // G: Total = Price * Quantity
    calcSheet.getRange("G" + i).setFormula(
      `=IF(AND(ISNUMBER(E${i}), ISNUMBER(F${i})), E${i} * F${i}, 0)`
    );
  }

  // 5. Formatting
  calcSheet.getRange("E2:E100").setNumberFormat("#,##0.00 [$€]"); // Currency
  calcSheet.getRange("G2:G100").setNumberFormat("#,##0.00 [$€]"); // Currency
  calcSheet.autoResizeColumns(1, 8);
  calcSheet.setColumnWidth(2, 300); // Make 'Work Item' column wider
  calcSheet.setColumnWidth(3, 400); // Make 'Description' column wider
  
  // 6. Summary Section
  var summaryStartRow = 102;
  calcSheet.getRange("A" + summaryStartRow).setValue("SUMMARY").setFontWeight("bold").setFontSize(14);
  calcSheet.getRange("A" + (summaryStartRow + 1) + ":B" + (summaryStartRow + 1)).setValues([["Subtotal", `=SUM(G2:G100)`]]);
  calcSheet.getRange("A" + (summaryStartRow + 2) + ":B" + (summaryStartRow + 2)).setValues([["Discount (%)", "0"]]);
  calcSheet.getRange("A" + (summaryStartRow + 3) + ":B" + (summaryStartRow + 3)).setValues([["Discount Amount", `=B${summaryStartRow + 1} * B${summaryStartRow + 2} / 100`]]);
  calcSheet.getRange("A" + (summaryStartRow + 4) + ":B" + (summaryStartRow + 4)).setValues([["Total After Discount", `=B${summaryStartRow + 1} - B${summaryStartRow + 3}`]]);
  calcSheet.getRange("B" + (summaryStartRow + 1) + ":B" + (summaryStartRow + 4)).setNumberFormat("#,##0.00 [$€]");
  calcSheet.getRange("B" + (summaryStartRow + 2)).setNumberFormat("0.00%");
  calcSheet.getRange("A" + summaryStartRow + ":B" + (summaryStartRow + 4)).setBackground("#e6f3ff");

  // 7. Add a button or instruction for template loading (future enhancement)
  calcSheet.getRange("A" + (summaryStartRow + 6)).setValue("TEMPLATE LOADER (Run 'Load Repair Template' from menu)").setFontWeight("bold");
  
  SpreadsheetApp.getUi().alert("Enhanced Constructor created! Go to 'Estimate_Calculator' tab.");
}

// Function to load a predefined template for repairs
function loadRepairTemplate() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var calcSheet = ss.getSheetByName("Estimate_Calculator");
  var dbSheet = ss.getSheetByName("DB_Works");
  
  if (!calcSheet || !dbSheet) {
    SpreadsheetApp.getUi().alert("Error: Required sheets not found.");
    return;
  }
  
  // Clear existing data (rows 2 to 50 for example)
  calcSheet.getRange("A2:H50").clearContent();
  
  // Define a simple template (this could be expanded or read from another sheet)
  var templateWorks = [
    ["Repairs", "Wall Painting", "", "", "", 20, "", "Living Room"],
    ["Repairs", "Floor Tiling", "", "", "", 10, "", "Kitchen"],
    ["Repairs", "Plumbing Repair", "", "", "", 1, "", "Bathroom"]
  ];
  
  // Insert template data starting from row 2
  calcSheet.getRange(2, 1, templateWorks.length, templateWorks[0].length).setValues(templateWorks);
  
  SpreadsheetApp.getUi().alert("Repair template loaded!");
}

// Add menu for easy access
function onOpen() {
  var ui = SpreadsheetApp.getUi();
  ui.createMenu('Estimate Tools')
    .addItem('Setup Constructor', 'setupEstimateConstructor')
    .addItem('Load Repair Template', 'loadRepairTemplate')
    .addToUi();
}
