/**
 * ESTIMATE CONSTRUCTOR SCRIPT
 * 
 * usage:
 * 1. Ensure you have a sheet named 'DB_Works' with columns: Category, Name, Unit, Price, Description
 * 2. Run the 'setupEstimateConstructor' function.
 */

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
  // We assume DB_Works data starts at row 2. Column B is 'Name'.
  var lastDbRow = dbSheet.getLastRow();
  if (lastDbRow > 1) {
    var workNamesRange = dbSheet.getRange(2, 2, lastDbRow - 1, 1);
    var rule = SpreadsheetApp.newDataValidation()
      .requireValueInRange(workNamesRange)
      .setAllowInvalid(false)
      .setHelpText("Select a valid work item from the database.")
      .build();
      
    // Apply dropdown to Column B (Work Item) for 100 rows
    calcSheet.getRange("B2:B100").setDataValidation(rule);
  }

  // 4. Add Formulas
  // We will use ARRAYFORMULA in row 2 so they auto-expand, or simple row-by-row formulas.
  // For reliability, let's set formulas for the first 100 rows.
  
  // Column A (Category): =IFNA(VLOOKUP(B2, DB_Works!B:E, 1, FALSE), "") - Note: DB col order matters.
  // Actually, typical VLOOKUP needs the search key to be in the first column of the range.
  // Let's just do a simpler approach: User picks Item (Col B), we lookup everything else.
  
  // We need to handle the VLOOKUP range. If DB_Works is: A=Cat, B=Name, C=Unit, D=Price
  // We need a helper column or we just use INDEX/MATCH.
  // Let's use standard cell formulas for rows 2-100.
  
  for (var i = 2; i <= 100; i++) {
    // A: Category (Index/Match to find Category (A) based on Name (B))
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
  
  SpreadsheetApp.getUi().alert("Constructor created! Go to 'Estimate_Calculator' tab.");
}
