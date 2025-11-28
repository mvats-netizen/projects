/**
 * =======================================
 * COURSE EMAIL GENERATOR - SECURE VERSION
 * =======================================
 * 
 * AI-powered email generation for Coursera courses
 * - Supports single and bulk processing
 * - Includes comprehensive access logging
 * - Uses secure DataBridge library for all business logic
 * 
 * SECURITY ARCHITECTURE:
 * All sensitive components are in the secure DataBridge library:
 * - API key management (Gemini, Databricks)
 * - Databricks data querying logic
 * - Email prompt generation
 * - Gemini AI integration
 * 
 * SETUP REQUIREMENTS:
 * 1. Create a separate Apps Script project (api.js)
 * 2. Add it as a library named "DataBridge" 
 * 3. The library provides these secure functions:
 *    - API Key Management: getGeminiApiKey(), getDbAccessToken(), getWarehouseId(), getServer()
 *    - Data Access: getCourseDataFull(courseSlug)
 *    - AI Integration: callGemini(promptText), generateEmailPrompts(course, userName, tone)
 */

/**
 * ===============================
 * CONFIGURATION
 * ===============================
 */

/**
 * Test API keys from library - Run this to verify your DataBridge setup
 */


/**
 * ===============================
 * CORE API FUNCTIONS
 * ===============================
 */

// Gemini API call function moved to secure DataBridge

/**
 * ===============================
 * DATABRICKS DATA FUNCTIONS
 * ===============================
 */

/**
 * Get course data from Databricks (via secure DataBridge)
 */
function getCourseData(courseSlug) {
    Logger.log(`🚀 getCourseData for: ${courseSlug}`);
    return DataBridge.getCourseDataFull(courseSlug);
  }
  
  // Full course data query moved to secure DataBridge
  
  /**
   * ===============================
   * EMAIL GENERATION
   * ===============================
   */
  
  /**
   * Generate emails using Gemini AI (via secure DataBridge)
   */
  function generateEmailsWithGemini(course, userName, tone = 'Professional') {
    // Debug logging
    Logger.log('🎯 Email generation starting:');
    Logger.log('Course modules received: ' + JSON.stringify(course.modules_json));
    Logger.log('Modules count: ' + (course.modules_json ? course.modules_json.length : 0));
    
    // Get all prompts from secure library
    const prompts = DataBridge.generateEmailPrompts(course, userName, tone);
    
    Logger.log('📝 Prompts generated: ' + prompts.length);
    Logger.log('Prompt types: ' + JSON.stringify(prompts.map(p => p.type)));
    
    // Generate emails for each prompt
    const emails = [];
    
    for (const promptObj of prompts) {
      try {
        const content = DataBridge.callGemini(promptObj.prompt);
        emails.push({ type: promptObj.type, content: content });
      } catch (error) {
        Logger.log(`Error generating email for ${promptObj.type}: ${error.message}`);
        throw error;
      }
    }
  
    return emails;
  }
  
  /**
   * ===============================
   * MAIN PROCESSING FUNCTIONS
   * ===============================
   */
  
  /**
   * Process single course from cells and generate emails
   */
  function processFromCells() {
    const startTime = Date.now();
    const ss = SpreadsheetApp.getActiveSpreadsheet();
    const sheet = ss.getActiveSheet();
    
    // Email generation logging will be handled at the end
    
    try {
      SpreadsheetApp.getActiveSpreadsheet().toast("🚀 Starting email generation process...", "Processing", 3);
      
      // Update dynamic processing time cell
      sheet.getRange("A16").setValue("🚀 Processing started • Reading inputs...");
      sheet.getRange("A16").setBackground("#FFF3CD").setFontColor("#856404");
      SpreadsheetApp.flush();
      
      // Read input from cells
      const courseSlug = sheet.getRange("B12").getValue().toString().trim();
      const emailTone = sheet.getRange("B13").getValue().toString().trim();
      
      SpreadsheetApp.getActiveSpreadsheet().toast("🔍 Validating course information...", "Validating", 2);
      
      // Validate inputs
      if (!courseSlug || courseSlug === "Enter the course slug here (e.g., machine-learning)") {
        SpreadsheetApp.getActiveSpreadsheet().toast("❌ Please enter a valid course slug", "Error", 4);
        throw new Error("Please enter a valid course slug");
      }
      
      // Set default tone if not selected
      const selectedTone = ['Professional', 'Motivational', 'Descriptive'].includes(emailTone) ? emailTone : 'Professional';
      
      // Use %NAME% placeholder
      const userName = "%NAME%";
      
      Logger.log(`Processing: ${courseSlug} with tone: ${selectedTone} (using %NAME% placeholder)`);
      
      SpreadsheetApp.getActiveSpreadsheet().toast(`📚 Fetching course data for: ${courseSlug}`, "Loading", 2);
      
      // Update processing cell for data loading
      sheet.getRange("A16").setValue(`📚 Loading course data: ${courseSlug}...`);
      SpreadsheetApp.flush();
      
      // Get course data
      const course = getCourseData(courseSlug);
      
      SpreadsheetApp.getActiveSpreadsheet().toast(`🤖 Generating ${selectedTone.toLowerCase()} emails with AI... ⏰ This takes 30 sec to 1 min`, "Generating", 4);
      
      // Update processing cell for AI generation
      sheet.getRange("A16").setValue(`🤖 Generating ${selectedTone.toLowerCase()} emails for: ${course.course_name}...`);
      sheet.getRange("A16").setBackground("#E3F2FD").setFontColor("#1565C0");
      SpreadsheetApp.flush();
      
      // Generate emails with selected tone
      const emails = generateEmailsWithGemini(course, userName, selectedTone);
      
      // Update processing cell for writing results
      sheet.getRange("A16").setValue("📝 Writing results to sheet...");
      sheet.getRange("A16").setBackground("#FFF3CD").setFontColor("#856404");
      SpreadsheetApp.flush();
      
      // Clear previous results (from row 19 onwards)
      const lastRow = sheet.getLastRow();
      if (lastRow >= 19) {
        sheet.getRange(19, 1, lastRow - 18, 6).clear();
      }
      
      // Prepare output data (6-column layout with merged content)
      const outputData = [];
      emails.forEach((email, index) => {
        outputData.push([
          course.course_name, 
          email.type, 
          email.content,  // Keep full content in column C
          "",             // Empty columns D and E (will be merged with C)
          "",
          "✅ Generated"
        ]);
      });
      
      SpreadsheetApp.getActiveSpreadsheet().toast("📝 Writing results to sheet...", "Finalizing", 2);
      
      // Write results starting from row 19
      if (outputData.length > 0) {
        const outputRange = sheet.getRange(19, 1, outputData.length, 6);
        outputRange.setValues(outputData);
        
        // Format with beautiful Coursera colors and merge content columns
        for (let i = 0; i < outputData.length; i++) {
          const rowRange = sheet.getRange(19 + i, 1, 1, 6);
          if (i % 2 === 0) {
            rowRange.setBackground("#FFFFFF");
          } else {
            rowRange.setBackground("#F8F9FA");
          }
          rowRange.setVerticalAlignment("top")
            .setWrap(true)
            .setBorder(false, false, true, false, false, false, "#DEE2E6", SpreadsheetApp.BorderStyle.SOLID);
          
          // Set Course and Email Type columns to middle alignment
          sheet.getRange(19 + i, 1).setVerticalAlignment("middle");
          sheet.getRange(19 + i, 2).setVerticalAlignment("middle");
          
          // Merge content columns (C, D, E) for continuous email display
          const contentRange = sheet.getRange(19 + i, 3, 1, 3);
          contentRange.merge();
        }
        
        // Highlight status column with blue theme
        sheet.getRange(19, 6, outputData.length, 1)
          .setBackground("#E7F3FF")
          .setFontWeight("bold")
          .setFontColor("#0056D3")
          .setHorizontalAlignment("center");
      }
      
      // Update processing cell to show completion
      sheet.getRange("A16").setValue(`🎉 COMPLETED! Generated ${emails.length} ${selectedTone.toLowerCase()} emails for: ${course.course_name}`);
      sheet.getRange("A16").setBackground("#E8F5E8").setFontColor("#2E7D32");
      SpreadsheetApp.flush();
      
      // Show success status via toast
      SpreadsheetApp.getActiveSpreadsheet().toast(`🎉 Successfully generated ${emails.length} emails with ${selectedTone.toLowerCase()} tone using %NAME% placeholder!`, "Success", 5);
      
      Logger.log(`Successfully generated ${emails.length} emails for: ${course.course_name}`);
      
      // Log successful email generation
      try {
        const processingTime = Date.now() - startTime;
        if (typeof logSingleEmailGeneration === 'function') {
          logSingleEmailGeneration(courseSlug, selectedTone, emails.length, processingTime, true);
        }
      } catch (error) {
        Logger.log(`⚠️ Email generation logging error: ${error.message}`);
      }
      
      return `✅ Success! Generated ${emails.length} emails for "${course.course_name}"`;
      
    } catch (error) {
      Logger.log("Error in processFromCells:", error.message);
      
      // Log failed email generation
      try {
        const processingTime = Date.now() - startTime;
        const courseSlug = sheet.getRange("B12").getValue().toString().trim();
        const emailTone = sheet.getRange("B13").getValue().toString().trim() || 'Professional';
        if (typeof logSingleEmailGeneration === 'function') {
          logSingleEmailGeneration(courseSlug, emailTone, 0, processingTime, false, error.message);
        }
      } catch (logError) {
        Logger.log(`⚠️ Email generation logging error: ${logError.message}`);
      }
      
      // Update processing cell for error
      sheet.getRange("A16").setValue(`❌ PROCESSING FAILED: ${error.message}`);
      sheet.getRange("A16").setBackground("#FFEBEE").setFontColor("#C62828");
      SpreadsheetApp.flush();
      
      // Show error via toast
      SpreadsheetApp.getActiveSpreadsheet().toast(`❌ Error: ${error.message}`, "Error", 5);
      
      throw error;
    }
  }
  
  /**
   * Process multiple course slugs and generate bulk email table
   */
  function processBulkCourses() {
    const startTime = Date.now();
    const ss = SpreadsheetApp.getActiveSpreadsheet();
    const bulkSheet = ss.getSheetByName("Bulk Email Generator");
    
    if (!bulkSheet) {
      throw new Error("Bulk Email Generator sheet not found. Please create it first.");
    }
    
    // Bulk email generation logging will be handled at the end
    
    try {
      SpreadsheetApp.getActiveSpreadsheet().toast("🚀 Starting bulk email generation process...", "Processing", 3);
      
      // Read inputs
      const emailTone = bulkSheet.getRange("B12").getValue().toString().trim();
      
      SpreadsheetApp.getActiveSpreadsheet().toast("🔍 Validating bulk course inputs...", "Validating", 2);
      
      // Set default tone if not selected
      const selectedTone = ['Professional', 'Motivational', 'Descriptive'].includes(emailTone) ? emailTone : 'Professional';
      
      // Use %NAME% placeholder
      const userName = "%NAME%";
      
      // Update dynamic processing time cell
      bulkSheet.getRange("A15").setValue("🚀 Processing started • Scanning course inputs...");
      bulkSheet.getRange("A15").setBackground("#FFF3CD").setFontColor("#856404");
      SpreadsheetApp.flush();
      
      // Get course slugs from column A (starting from row 19, up to 15 courses)
      const courseSlugs = [];
      for (let row = 19; row <= 33; row++) {
        const slug = bulkSheet.getRange(row, 1).getValue().toString().trim();
        if (!slug) continue;
        
        // Skip sample/placeholder slugs if real data exists
        if (slug !== "machine-learning" && slug !== "data-science-fundamentals" && slug !== "python-programming") {
          courseSlugs.push(slug);
        } else if (courseSlugs.length === 0) {
          // If no real slugs entered, process the samples
          courseSlugs.push(slug);
        }
      }
      
      if (courseSlugs.length === 0) {
        SpreadsheetApp.getActiveSpreadsheet().toast("❌ Please enter at least one course slug", "Error", 4);
        throw new Error("Please enter at least one course slug");
      }
      
      // Show number of courses to process with timing estimate
      const estimatedMinutes = courseSlugs.length;
      SpreadsheetApp.getActiveSpreadsheet().toast(`📚 Found ${courseSlugs.length} courses to process with ${selectedTone.toLowerCase()} tone ⏰ Est. ${estimatedMinutes} min`, "Loading", 3);
      
      Logger.log(`Processing ${courseSlugs.length} courses with tone: ${selectedTone} using %NAME% placeholder`);
      
      // Update dynamic processing time cell with course count
      bulkSheet.getRange("A15").setValue(`📚 Found ${courseSlugs.length} courses • Starting bulk processing...`);
      bulkSheet.getRange("A15").setBackground("#E3F2FD").setFontColor("#1565C0");
      SpreadsheetApp.flush();
      
      // Clear previous results (from row 37 onwards)
      const lastRow = bulkSheet.getLastRow();
      if (lastRow >= 37) {
        bulkSheet.getRange(37, 1, lastRow - 36, 6).clear();
      }
      
      // Process each course and collect results
      const allResults = [];
      let processedCount = 0;
      
      for (const slug of courseSlugs) {
        try {
          // Show progress
          SpreadsheetApp.getActiveSpreadsheet().toast(`⏳ Processing ${slug} (${processedCount + 1}/${courseSlugs.length}) with ${selectedTone.toLowerCase()} tone... ⏰ ~1 min per course`, "Processing", 3);
          
          // Update dynamic processing time cell for current course
          bulkSheet.getRange("A15").setValue(`⏳ Processing course ${processedCount + 1}/${courseSlugs.length}: ${slug}...`);
          bulkSheet.getRange("A15").setBackground("#E3F2FD").setFontColor("#1565C0");
          SpreadsheetApp.flush();
          
          // Get course data
          const course = getCourseData(slug);
          
          // Generate emails with selected tone
          const emails = generateEmailsWithGemini(course, userName, selectedTone);
          
          // Add to results (6-column layout with merged content)
          emails.forEach(email => {
            allResults.push([
              course.course_name,
              email.type,
              email.content,  // Keep full content in column C
              "",             // Empty columns D and E (will be merged with C)
              "",
              "✅ Generated"
            ]);
          });
          
          processedCount++;
          Logger.log(`Successfully processed: ${slug}`);
          
        } catch (error) {
          Logger.log(`Error processing ${slug}: ${error.message}`);
          
          // Update dynamic processing time cell for error
          bulkSheet.getRange("A15").setValue(`❌ Error processing ${slug}: ${error.message}`);
          bulkSheet.getRange("A15").setBackground("#FFEBEE").setFontColor("#C62828");
          SpreadsheetApp.flush();
          
          allResults.push([
            slug,
            "Error",
            `Failed to process: ${error.message}`,  // Keep full error in column C
            "",                                     // Empty columns D and E (will be merged with C)
            "",
            "❌ Error"
          ]);
        }
      }
      
      SpreadsheetApp.getActiveSpreadsheet().toast("📝 Writing bulk results to sheet...", "Finalizing", 2);
      
      // Update dynamic processing time cell for writing results
      bulkSheet.getRange("A15").setValue("📝 Writing all results to sheet...");
      bulkSheet.getRange("A15").setBackground("#FFF3CD").setFontColor("#856404");
      SpreadsheetApp.flush();
      
      // Write all results (starting from row 37) - 6 columns
      if (allResults.length > 0) {
        const outputRange = bulkSheet.getRange(37, 1, allResults.length, 6);
        outputRange.setValues(allResults);
        
        // Style the results
        outputRange.setBorder(true, true, true, true, true, true, "#0056D3", SpreadsheetApp.BorderStyle.SOLID);
        outputRange.setVerticalAlignment("top");
        outputRange.setWrap(true);
        
        // Alternate row colors and merge content columns
        for (let i = 0; i < allResults.length; i++) {
          const rowRange = bulkSheet.getRange(37 + i, 1, 1, 6);
          if (i % 2 === 0) {
            rowRange.setBackground("#FFFFFF");
          } else {
            rowRange.setBackground("#F8F9FA");
          }
          
          // Set Course and Email Type columns to middle alignment
          bulkSheet.getRange(37 + i, 1).setVerticalAlignment("middle");
          bulkSheet.getRange(37 + i, 2).setVerticalAlignment("middle");
          
          // Merge content columns (C, D, E) for continuous email display
          const contentRange = bulkSheet.getRange(37 + i, 3, 1, 3);
          contentRange.merge();
          
          // Color code status column (column 6 - last column)
          const statusCell = bulkSheet.getRange(37 + i, 6);
          const status = allResults[i][5];
          if (status === "✅ Generated") {
            statusCell.setBackground("#D4EDDA").setFontColor("#155724");
          } else if (status === "❌ Error") {
            statusCell.setBackground("#F8D7DA").setFontColor("#721C24");
          }
        }
      }
      
      // Update dynamic processing time cell to show completion
      bulkSheet.getRange("A15").setValue(`🎉 COMPLETED! Processed ${processedCount}/${courseSlugs.length} courses • Generated ${allResults.length} emails total`);
      bulkSheet.getRange("A15").setBackground("#E8F5E8").setFontColor("#2E7D32");
      SpreadsheetApp.flush();
      
      // Show completion message via toast
      SpreadsheetApp.getActiveSpreadsheet().toast(`✅ Generated ${allResults.length} ${selectedTone.toLowerCase()} emails for ${processedCount} courses using %NAME% placeholder!`, "Bulk Processing Complete", 5);
      
      Logger.log(`Bulk processing complete: ${processedCount}/${courseSlugs.length} courses, ${allResults.length} emails generated with ${selectedTone} tone using %NAME% placeholder`);
      
      // Log bulk email generation completion
      try {
        const totalTime = Date.now() - startTime;
        const failureCount = courseSlugs.length - processedCount;
        if (typeof logBulkEmailGeneration === 'function') {
          logBulkEmailGeneration(courseSlugs, selectedTone, allResults.length, totalTime, processedCount, failureCount);
        }
      } catch (error) {
        Logger.log(`⚠️ Email generation logging error: ${error.message}`);
      }
      
    } catch (error) {
      Logger.log(`Bulk processing error: ${error.message}`);
      
      // Log failed bulk email generation
      try {
        const totalTime = Date.now() - startTime;
        const emailTone = bulkSheet.getRange("B12").getValue().toString().trim() || 'Professional';
        if (typeof logBulkEmailGeneration === 'function') {
          logBulkEmailGeneration(['Failed'], emailTone, 0, totalTime, 0, 1, error.message);
        }
      } catch (logError) {
        Logger.log(`⚠️ Email generation logging error: ${logError.message}`);
      }
      
      // Update dynamic processing time cell for general error
      const bulkSheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName("Bulk Email Generator");
      if (bulkSheet) {
        bulkSheet.getRange("A15").setValue(`❌ PROCESSING FAILED: ${error.message}`);
        bulkSheet.getRange("A15").setBackground("#FFEBEE").setFontColor("#C62828");
        SpreadsheetApp.flush();
      }
      
      SpreadsheetApp.getActiveSpreadsheet().toast(`❌ Error: ${error.message}`, "Bulk Processing Error", 5);
      throw error;
    }
  }
  
  /**
   * ===============================
   * SHEET SETUP FUNCTIONS
   * ===============================
   */
  
  /**
   * Sets up a beautiful Coursera-inspired main sheet template
   */
  function setupInputTemplate() {
    const ss = SpreadsheetApp.getActiveSpreadsheet();
    const sheet = ss.getActiveSheet();
    
    // Clear the sheet first
    sheet.clear();
    
    // Coursera color palette
    const courseraBlue = "#0056D3";
    const lightBlue = "#E7F3FF";
    const darkBlue = "#003D99";
    
    // Main header (Full Width - 6 COLUMNS)
    sheet.getRange("A1:F1").merge().setValue("📚 COURSERA EMAIL GENERATOR");
    sheet.getRange("A1").setBackground(courseraBlue)
      .setFontColor("white")
      .setFontWeight("bold")
      .setFontSize(18)
      .setHorizontalAlignment("center")
      .setVerticalAlignment("middle");
    sheet.setRowHeight(1, 50);
    
    // Subtitle
    sheet.getRange("A2:F2").merge().setValue("AI-Powered Course Email Generation");
    sheet.getRange("A2").setBackground(lightBlue)
      .setFontColor(darkBlue)
      .setFontStyle("italic")
      .setFontSize(12)
      .setHorizontalAlignment("center")
      .setVerticalAlignment("middle");
    sheet.setRowHeight(2, 35);
    
    // Spacing
    sheet.setRowHeight(3, 20);
    
    // HOW IT WORKS SECTION
    sheet.getRange("A4:F4").merge().setValue("📖  HOW IT WORKS");
    sheet.getRange("A4").setBackground(lightBlue)
      .setFontColor(darkBlue)
      .setFontWeight("bold")
      .setFontSize(16)
      .setHorizontalAlignment("center")
      .setVerticalAlignment("middle")
      .setBorder(true, true, true, true, false, false, courseraBlue, SpreadsheetApp.BorderStyle.SOLID_MEDIUM);
    sheet.setRowHeight(4, 40);
    
    // Instructions
    const instructions = [
      "1️⃣  Enter the course slug (e.g., machine-learning)",
      "2️⃣  Select email tone: Professional, Motivational, or Descriptive", 
      "3️⃣  Use menu: '📧 Course Email Generator' → '🚀 Generate Single Course Emails'",
      "4️⃣  Review your customized email sequence (uses %NAME% placeholder)",
      "⏱️  Processing time: 30 seconds to 1 minute per course"
    ];
    
    for (let i = 0; i < instructions.length; i++) {
      sheet.getRange(5 + i, 1, 1, 6).merge().setValue(instructions[i]);
      sheet.getRange(5 + i, 1).setBackground("#F8F9FA")
        .setFontColor("#495057")
        .setVerticalAlignment("middle")
        .setHorizontalAlignment("left")
        .setBorder(true, true, true, true, false, false, "#E9ECEF", SpreadsheetApp.BorderStyle.SOLID);
      sheet.setRowHeight(5 + i, 32);
    }
    
    // Spacing after instructions
    sheet.setRowHeight(10, 25);
  
    // INPUT SECTION
    sheet.getRange("A11:F11").merge().setValue("✏️  COURSE DETAILS");
    sheet.getRange("A11").setBackground(courseraBlue)
      .setFontColor("white")
      .setFontWeight("bold")
      .setFontSize(16)
      .setHorizontalAlignment("center")
      .setVerticalAlignment("middle")
      .setBorder(true, true, true, true, false, false, courseraBlue, SpreadsheetApp.BorderStyle.SOLID_MEDIUM);
    sheet.setRowHeight(11, 40);
  
    // Input fields
    const inputData = [
      ["🔗 Course Slug", "Enter the course slug here (e.g., machine-learning)"],
      ["🎯 Email Tone", "Professional"]
    ];
    
    sheet.getRange(12, 1, inputData.length, 2).setValues(inputData);
    
    // Style input labels
    sheet.getRange("A12:A13").setBackground(lightBlue)
      .setFontColor(darkBlue)
      .setFontWeight("bold")
      .setVerticalAlignment("middle")
      .setHorizontalAlignment("left")
      .setBorder(true, true, true, true, false, false, "#B3D9FF", SpreadsheetApp.BorderStyle.SOLID);
    
    // Style input fields
    sheet.getRange("B12:F13").setBackground("white")
      .setFontColor("#333333")
      .setBorder(true, true, true, true, false, false, "#B3D9FF", SpreadsheetApp.BorderStyle.SOLID)
      .setVerticalAlignment("middle");
    
    // Add dropdown validation for email tone
    const toneValidation = SpreadsheetApp.newDataValidation()
      .requireValueInList(['Professional', 'Motivational', 'Descriptive'], true)
      .setAllowInvalid(false)
      .setHelpText('Select the tone for generated emails: Professional (formal), Motivational (inspiring), or Descriptive (detailed)')
      .build();
    
    sheet.getRange("B13").setDataValidation(toneValidation);
    sheet.getRange("B13").setValue("Professional");
    
    sheet.setRowHeight(12, 35);
    sheet.setRowHeight(13, 35);
    
    // Spacing
    sheet.setRowHeight(14, 20);
    
    // Dynamic status section
    sheet.getRange("A15:F15").merge().setValue("💤 Ready to process • Click 'Generate Single Course Emails' in menu to start");
    sheet.getRange("A15").setBackground("#E8F5E8")
      .setFontColor("#2E7D32")
      .setFontWeight("bold")
      .setFontSize(14)
      .setHorizontalAlignment("center")
      .setVerticalAlignment("middle")
      .setBorder(true, true, true, true, false, false, "#4CAF50", SpreadsheetApp.BorderStyle.SOLID_MEDIUM);
    sheet.setRowHeight(15, 35);
    
    // Processing time info
    sheet.getRange("A16:F16").merge().setValue("⏰ PROCESSING TIME: 30 seconds to 1 minute per course");
    sheet.getRange("A16").setBackground("#FFF3CD")
      .setFontColor("#856404")
      .setFontWeight("bold")
      .setFontSize(14)
      .setHorizontalAlignment("center")
      .setVerticalAlignment("middle")
      .setBorder(true, true, true, true, false, false, "#FFEAA7", SpreadsheetApp.BorderStyle.SOLID_MEDIUM);
    sheet.setRowHeight(16, 35);
    
    // Spacing
    sheet.setRowHeight(17, 20);
    
    // RESULTS SECTION
    sheet.getRange("A18:F18").merge().setValue("📊  GENERATED EMAIL SEQUENCE");
    sheet.getRange("A18").setBackground(courseraBlue)
      .setFontColor("white")
      .setFontWeight("bold")
      .setFontSize(16)
      .setHorizontalAlignment("center")
      .setVerticalAlignment("middle")
      .setBorder(true, true, true, true, false, false, courseraBlue, SpreadsheetApp.BorderStyle.SOLID_MEDIUM);
    sheet.setRowHeight(18, 40);
    
    // Headers
    const outputHeaders = ["Course", "Email Type", "Content", "", "", "Status"];
    sheet.getRange(19, 1, 1, 6).setValues([outputHeaders]);
    sheet.getRange("A19:F19").setBackground(lightBlue)
      .setFontColor(darkBlue)
      .setFontWeight("bold")
      .setHorizontalAlignment("center")
      .setVerticalAlignment("middle")
      .setBorder(true, true, true, true, false, false, "#B3D9FF", SpreadsheetApp.BorderStyle.SOLID);
    sheet.setRowHeight(19, 32);
    
    // Column formatting
    sheet.setColumnWidth(1, 150); // Course
    sheet.setColumnWidth(2, 120); // Email Type  
    sheet.setColumnWidth(3, 300); // Content (main column)
    sheet.setColumnWidth(4, 100); // Content overflow
    sheet.setColumnWidth(5, 100); // Content overflow  
    sheet.setColumnWidth(6, 80);  // Status
    
    // Outer border
    sheet.getRange("A1:F19").setBorder(true, true, true, true, false, false, courseraBlue, SpreadsheetApp.BorderStyle.SOLID_MEDIUM);
    
    // Freeze headers
    sheet.setFrozenRows(2);
    
    Logger.log("Beautiful Coursera-themed template created!");
    return "✅ Coursera-style template ready!";
  }
  
  /**
   * Set up bulk processing template for multiple course slugs
   */
  function setupBulkTemplate() {
    try {
      const ss = SpreadsheetApp.getActiveSpreadsheet();
      
      // Create or get bulk processing sheet
      let bulkSheet = ss.getSheetByName("Bulk Email Generator");
      
      if (bulkSheet) {
        Logger.log("Bulk sheet exists, clearing content...");
        bulkSheet.clear();
      } else {
        Logger.log("Creating new bulk sheet...");
        bulkSheet = ss.insertSheet("Bulk Email Generator");
      }
      
      // Coursera color palette
      const courseraBlue = "#0056D3";
      const lightBlue = "#E7F3FF";
      const darkBlue = "#003D99";
    
      // Main header
      bulkSheet.getRange("A1:F1").merge().setValue("📚 BULK EMAIL GENERATOR");
      bulkSheet.getRange("A1").setBackground(courseraBlue)
        .setFontColor("white")
        .setFontWeight("bold")
        .setFontSize(18)
        .setHorizontalAlignment("center")
        .setVerticalAlignment("middle");
      bulkSheet.setRowHeight(1, 50);
      
      // Subtitle
      bulkSheet.getRange("A2:F2").merge().setValue("AI-Powered Bulk Course Email Generation");
      bulkSheet.getRange("A2").setBackground(lightBlue)
        .setFontColor(darkBlue)
        .setFontStyle("italic")
        .setFontSize(12)
        .setHorizontalAlignment("center")
        .setVerticalAlignment("middle");
      bulkSheet.setRowHeight(2, 35);
      
      // Spacing
      bulkSheet.setRowHeight(3, 20);
      
      // HOW IT WORKS SECTION
      bulkSheet.getRange("A4:F4").merge().setValue("📖  HOW IT WORKS");
      bulkSheet.getRange("A4").setBackground(lightBlue)
        .setFontColor(darkBlue)
        .setFontWeight("bold")
        .setFontSize(16)
        .setHorizontalAlignment("center")
        .setVerticalAlignment("middle")
        .setBorder(true, true, true, true, false, false, courseraBlue, SpreadsheetApp.BorderStyle.SOLID_MEDIUM);
      bulkSheet.setRowHeight(4, 40);
      
      // Instructions
      const bulkInstructions = [
        "1️⃣  Select email tone: Professional, Motivational, or Descriptive",
        "2️⃣  Add course slugs (e.g., machine-learning) in column A below (up to 15 courses)", 
        "3️⃣  Use menu: '📧 Course Email Generator' → '📊 Process Multiple Courses'",
        "4️⃣  Review your customized email sequences (uses %NAME% placeholder)",
        "⏱️  Processing time: 30 seconds to 1 minute per course"
      ];
      
      for (let i = 0; i < bulkInstructions.length; i++) {
        bulkSheet.getRange(5 + i, 1, 1, 6).merge().setValue(bulkInstructions[i]);
        bulkSheet.getRange(5 + i, 1).setBackground("#F8F9FA")
          .setFontColor("#495057")
          .setVerticalAlignment("middle")
          .setHorizontalAlignment("left")
          .setBorder(true, true, true, true, false, false, "#E9ECEF", SpreadsheetApp.BorderStyle.SOLID);
        bulkSheet.setRowHeight(5 + i, 32);
      }
      
      // Spacing after instructions
      bulkSheet.setRowHeight(10, 25);
  
      // INPUT SECTION
      bulkSheet.getRange("A11:F11").merge().setValue("✏️  BULK COURSE DETAILS");
      bulkSheet.getRange("A11").setBackground(courseraBlue)
        .setFontColor("white")
        .setFontWeight("bold")
        .setFontSize(16)
        .setHorizontalAlignment("center")
        .setVerticalAlignment("middle")
        .setBorder(true, true, true, true, false, false, courseraBlue, SpreadsheetApp.BorderStyle.SOLID_MEDIUM);
      bulkSheet.setRowHeight(11, 40);
      
      // Email tone selector
      bulkSheet.getRange("A12").setValue("🎯 Email Tone");
      bulkSheet.getRange("B12").setValue("Professional");
      
      // Style input label
      bulkSheet.getRange("A12").setBackground(lightBlue)
        .setFontColor(darkBlue)
        .setFontWeight("bold")
        .setVerticalAlignment("middle")
        .setHorizontalAlignment("left")
        .setBorder(true, true, true, true, false, false, "#B3D9FF", SpreadsheetApp.BorderStyle.SOLID);
      
      // Style ONLY the dropdown cell (B12)
      bulkSheet.getRange("B12").setBackground("white")
        .setFontColor("#333333")
        .setBorder(true, true, true, true, false, false, "#0056D3", SpreadsheetApp.BorderStyle.SOLID_MEDIUM)
        .setVerticalAlignment("middle")
        .setHorizontalAlignment("center");
      
      // Style remaining cells
      bulkSheet.getRange("C12:F12").setBackground("#F8F9FA")
        .setFontColor("#6C757D")
        .setBorder(true, true, true, true, false, false, "#E9ECEF", SpreadsheetApp.BorderStyle.SOLID)
        .setVerticalAlignment("middle");
      
      // Add dropdown validation ONLY to B12
      const bulkToneValidation = SpreadsheetApp.newDataValidation()
        .requireValueInList(['Professional', 'Motivational', 'Descriptive'], true)
        .setAllowInvalid(false)
        .setHelpText('Select the tone for generated emails: Professional (formal), Motivational (inspiring), or Descriptive (detailed)')
        .build();
      
      bulkSheet.getRange("B12").setDataValidation(bulkToneValidation);
      
      bulkSheet.setRowHeight(12, 35);
      
      // Spacing
      bulkSheet.setRowHeight(13, 20);
      
      // Dynamic status section
      bulkSheet.getRange("A14:F14").merge().setValue("💤 Ready to process • Click 'Process Multiple Courses' in menu to start");
      bulkSheet.getRange("A14").setBackground("#E8F5E8")
        .setFontColor("#2E7D32")
        .setFontWeight("bold")
        .setFontSize(14)
        .setHorizontalAlignment("center")
        .setVerticalAlignment("middle")
        .setBorder(true, true, true, true, false, false, "#4CAF50", SpreadsheetApp.BorderStyle.SOLID_MEDIUM);
      bulkSheet.setRowHeight(14, 35);
      
      // Processing time info
      bulkSheet.getRange("A15:F15").merge().setValue("⏰ PROCESSING TIME: ~1 minute per course • Multiple courses processed sequentially");
      bulkSheet.getRange("A15").setBackground("#FFF3CD")
        .setFontColor("#856404")
        .setFontWeight("bold")
        .setFontSize(14)
        .setHorizontalAlignment("center")
        .setVerticalAlignment("middle")
        .setBorder(true, true, true, true, false, false, "#FFEAA7", SpreadsheetApp.BorderStyle.SOLID_MEDIUM);
      bulkSheet.setRowHeight(15, 35);
      
      // Spacing
      bulkSheet.setRowHeight(16, 20);
      
      // COURSE SLUGS SECTION
      bulkSheet.getRange("A17:F17").merge().setValue("🔗  COURSE SLUGS");
      bulkSheet.getRange("A17").setBackground(courseraBlue)
        .setFontColor("white")
        .setFontWeight("bold")
        .setFontSize(16)
        .setHorizontalAlignment("center")
        .setVerticalAlignment("middle")
        .setBorder(true, true, true, true, false, false, courseraBlue, SpreadsheetApp.BorderStyle.SOLID_MEDIUM);
      bulkSheet.setRowHeight(17, 40);
      
      // Instructions for course slugs
      bulkSheet.getRange("A18:F18").merge().setValue("📝 Enter course slugs below (one per row). You can add up to 15 courses:");
      bulkSheet.getRange("A18").setBackground(lightBlue)
        .setFontColor(darkBlue)
        .setFontStyle("italic")
        .setFontSize(12)
        .setHorizontalAlignment("center")
        .setVerticalAlignment("middle")
        .setBorder(true, true, true, true, false, false, "#B3D9FF", SpreadsheetApp.BorderStyle.SOLID);
      bulkSheet.setRowHeight(18, 30);
      
      // Course slug input area (rows 19-33 = 15 courses)
      for (let i = 0; i < 15; i++) {
        const row = 19 + i;
        if (i < 3) {
          // Sample data for first 3 rows
          const sampleSlugs = ["machine-learning", "data-science-fundamentals", "python-programming"];
          bulkSheet.getRange(`A${row}`).setValue(sampleSlugs[i]);
          bulkSheet.getRange(`A${row}`).setBackground("#F8F9FA")
            .setFontStyle("italic")
            .setFontColor("#6C757D");
        } else {
          // Empty rows for additional courses
          bulkSheet.getRange(`A${row}`).setValue("");
          bulkSheet.getRange(`A${row}`).setBackground("white");
        }
        
        // Style all course input rows consistently
        bulkSheet.getRange(`A${row}`).setBorder(true, true, true, true, false, false, "#DEE2E6", SpreadsheetApp.BorderStyle.SOLID)
          .setVerticalAlignment("middle")
          .setHorizontalAlignment("left");
        bulkSheet.setRowHeight(row, 28);
      }
      
      // Spacing before results
      bulkSheet.setRowHeight(33, 25);
      
      // RESULTS SECTION
      bulkSheet.getRange("A34:F34").merge().setValue("📊  GENERATED EMAIL SEQUENCES");
      bulkSheet.getRange("A34").setBackground(courseraBlue)
        .setFontColor("white")
        .setFontWeight("bold")
        .setFontSize(16)
        .setHorizontalAlignment("center")
        .setVerticalAlignment("middle")
        .setBorder(true, true, true, true, false, false, courseraBlue, SpreadsheetApp.BorderStyle.SOLID_MEDIUM);
      bulkSheet.setRowHeight(34, 40);
      
      // Headers
      const outputHeaders = ["Course", "Email Type", "Content", "", "", "Status"];
      bulkSheet.getRange(35, 1, 1, 6).setValues([outputHeaders]);
      bulkSheet.getRange("A35:F35").setBackground(lightBlue)
        .setFontColor(darkBlue)
        .setFontWeight("bold")
        .setHorizontalAlignment("center")  
        .setVerticalAlignment("middle")
        .setBorder(true, true, true, true, false, false, "#B3D9FF", SpreadsheetApp.BorderStyle.SOLID);
      bulkSheet.setRowHeight(35, 32);
      
      // Column formatting
      bulkSheet.setColumnWidth(1, 150); // Course
      bulkSheet.setColumnWidth(2, 120); // Email Type  
      bulkSheet.setColumnWidth(3, 300); // Content (main column)
      bulkSheet.setColumnWidth(4, 100); // Content overflow
      bulkSheet.setColumnWidth(5, 100); // Content overflow  
      bulkSheet.setColumnWidth(6, 80);  // Status
      
      // Outer border
      bulkSheet.getRange("A1:F33").setBorder(true, true, true, true, false, false, courseraBlue, SpreadsheetApp.BorderStyle.SOLID_MEDIUM);
      
      // Freeze headers
      bulkSheet.setFrozenRows(2);
    
      SpreadsheetApp.getActiveSpreadsheet().toast("✅ Bulk email generator template created!", "Setup Complete", 3);
      Logger.log("Bulk processing template setup complete");
      
    } catch (error) {
      Logger.log("❌ Error setting up bulk template: " + error.message);
      Logger.log("❌ Error stack: " + error.stack);
      SpreadsheetApp.getActiveSpreadsheet().toast("❌ Setup failed: " + error.message, "Bulk Setup Error", 10);
      throw error;
    }
  }
  
  /**
   * ===============================
   * UTILITY FUNCTIONS
   * ===============================
   */
  
  /**
   * Reset inputs and results with Coursera styling
   */
  function resetCourseraInputs() {
    const ss = SpreadsheetApp.getActiveSpreadsheet();
    const sheet = ss.getActiveSheet();
    const sheetName = sheet.getName();
    
    try {
      SpreadsheetApp.getActiveSpreadsheet().toast("🔄 Resetting current sheet...", "Resetting", 2);
      
      if (sheetName === "Bulk Email Generator") {
        // Reset bulk sheet
        Logger.log("Resetting Bulk Email Generator sheet");
        
        // Reset email tone to default
        sheet.getRange("B12").setValue("Professional");
        
        // Clear all course slug inputs (rows 19-33)
        sheet.getRange("A19:A33").clear();
        
        // Add sample data for guidance
        const sampleSlugs = ["machine-learning", "data-science-fundamentals", "python-programming"];
        for (let i = 0; i < sampleSlugs.length; i++) {
          sheet.getRange(19 + i, 1).setValue(sampleSlugs[i]);
          sheet.getRange(19 + i, 1).setBackground("#F8F9FA")
            .setFontStyle("italic")
            .setFontColor("#6C757D");
        }
        
        // Reset dynamic processing time cell to default state
        sheet.getRange("A15:F15").merge().setValue("⏰ PROCESSING TIME: ~1 minute per course • Multiple courses processed sequentially");
        sheet.getRange("A15").setBackground("#FFF3CD")
          .setFontColor("#856404")
          .setFontWeight("bold")
          .setFontSize(14)
          .setHorizontalAlignment("center")
          .setVerticalAlignment("middle")
          .setBorder(true, true, true, true, false, false, "#FFEAA7", SpreadsheetApp.BorderStyle.SOLID_MEDIUM);
        
        // Clear previous results (from row 37 onwards) - 6 columns  
        const lastRow = sheet.getLastRow();
        if (lastRow >= 37) {
          sheet.getRange(37, 1, lastRow - 36, 6).clear();
        }
        
        SpreadsheetApp.getActiveSpreadsheet().toast("✅ Bulk sheet reset! Sample course slugs added", "Reset Complete", 4);
        
      } else {
        // Reset main sheet
        Logger.log("Resetting main EmailGenerator sheet");
        
        // Reset course slug to placeholder
        sheet.getRange("B12").setValue("Enter the course slug here (e.g., machine-learning)");
        sheet.getRange("B12").setBackground("white")
          .setFontStyle("italic")
          .setFontColor("#6C757D");
        
        // Reset email tone to default
        sheet.getRange("B13").setValue("Professional");
        
        // Reset dynamic processing time cell to default state
        sheet.getRange("A16:F16").merge().setValue("⏰ PROCESSING TIME: 30 seconds to 1 minute per course");
        sheet.getRange("A16").setBackground("#FFF3CD")
          .setFontColor("#856404")
          .setFontWeight("bold")
          .setFontSize(14)
          .setHorizontalAlignment("center")
          .setVerticalAlignment("middle")
          .setBorder(true, true, true, true, false, false, "#FFEAA7", SpreadsheetApp.BorderStyle.SOLID_MEDIUM);
        
        // Clear previous results (from row 19 onwards) - 6 columns
        const lastRow = sheet.getLastRow();
        if (lastRow >= 19) {
          sheet.getRange(19, 1, lastRow - 18, 6).clear();
        }
        
        SpreadsheetApp.getActiveSpreadsheet().toast("✅ Main sheet reset! Ready for new course input", "Reset Complete", 4);
      }
      
      Logger.log(`Successfully reset ${sheetName} sheet`);
      
    } catch (error) {
      Logger.log(`Error resetting ${sheetName} sheet:`, error.message);
      SpreadsheetApp.getActiveSpreadsheet().toast(`❌ Error resetting ${sheetName}: ${error.message}`, "Reset Error", 5);
    }
  }
  
  /**
   * Quick setup function - creates main sheet template
   */
  function quickSetup() {
    setupInputTemplate();
    
    const message = `
  🎉 Beautiful Coursera Template Created!
  
  🎨 What's been created:
  • Stunning Coursera-inspired blue and white design
  • Professional header with elegant styling
  • Clean input section for course slug and email tone
  • Step-by-step instructions with emoji guides
  • Professional results table
  • Real-time status updates with Coursera colors
  
  📝 How to use:
  1. Enter course slug in B12 (row 12)
  2. Select email tone in B13 (row 13)  
  3. Use menu: "📧 Course Email Generator" → "🚀 Generate Single Course Emails"
  4. View your beautiful email sequence below
  
  ✨ COURSERA FEATURES:
  • Beautiful blue and white color scheme
  • Professional typography and spacing
  • Menu-based operation (reliable and clean)
  • Clean, elegant design inspired by Coursera
  • Ultra-concise, professional emails
  • Status updates with beautiful colors
  • Uses %NAME% placeholder for universal compatibility
  `;
    
    Logger.log(message);
    SpreadsheetApp.getActiveSpreadsheet().toast("🎉 Beautiful Coursera template ready! Use menu to generate emails.", "Setup Complete", 6);
    
    return message;
  }
  
  /**
   * Show help information
   */
  function showHelp() {
    const ui = SpreadsheetApp.getUi();
    ui.alert('📚 STEP-BY-STEP GUIDE', 
      `🎯 SIMPLE MENU-BASED EMAIL GENERATOR
  
  ═══════════════════════════════════════════════════
  📋 SINGLE COURSE PROCESSING (Main Sheet)
  ═══════════════════════════════════════════════════
  
  STEP 1: Setup Template
  Menu → "📧 Course Email Generator" → "📋 Setup Main Sheet"
  
  STEP 2: Enter Course Information  
  • Course Slug (B12): Enter slug only (e.g., "machine-learning")
  • Email Tone (B13): Select Professional, Motivational, or Descriptive
  
  STEP 3: Generate Emails
  Menu → "📧 Course Email Generator" → "🚀 Generate Single Course Emails"
  
  STEP 4: Review Results
  Table Format: Course | Email Type | Content | Status
  • All emails use %NAME% placeholder for universal compatibility
  
  ═══════════════════════════════════════════════════
  📊 BULK PROCESSING (Multiple Courses) 
  ═══════════════════════════════════════════════════
  
  STEP 1: Setup Bulk Sheet
  Menu → "📧 Course Email Generator" → "📊 Setup Bulk Sheet"
  
  STEP 2: Enter Information
  • Email Tone (B12): Select Professional, Motivational, or Descriptive
  • Course Slugs (Column A, rows 16-30): One slug per row (up to 15 courses)
  
  STEP 3: Generate Bulk Emails
  Menu → "📧 Course Email Generator" → "📊 Process Multiple Courses"
  
  STEP 4: Review Bulk Results  
  Table Format: Course | Email Type | Content | Status
  • Same clean layout as single processing
  • Shows progress via toast notifications
  
  🎨 FEATURES:
  ✓ Clean, professional Coursera-inspired design
  ✓ Menu-based operation (no button clicking issues)
  ✓ Toast notifications for real-time feedback
  ✓ Tone selection for customized email style
  ✓ Uses %NAME% placeholder for universal use
  
  📧 EMAIL TYPES GENERATED:
  • Welcome Email: Course introduction
  • Module Emails: One per course module
  • Completion Email: Congratulations message
  • Announcement 1: Career value and market demand retention email
  • Announcement 2: Learning momentum and consistency habits email
  • Announcement 3: Real-world applications and industry opportunities retention email
  
  ⏱️ PROCESSING TIME:
  • Single course: 30 seconds to 1 minute
  • Multiple courses: 30 seconds to 1 minute per course
  
  🔧 CLEAN MENU OPTIONS:
  All functions accessible via "📧 Course Email Generator" menu
  • Simple, focused interface
  • No complex buttons or debugging tools
  • Just the essentials for email generation`, 
      ui.ButtonSet.OK);
  }
  
  /**
   * ===============================
   * MENU & EVENT HANDLERS
   * ===============================
   */
  
  /**
   * Create menu when sheet opens
   */
  function onOpen() {
    const ui = SpreadsheetApp.getUi();
    ui.createMenu('📧 Course Email Generator')
      .addItem('🚀 Generate Single Course Emails', 'processFromCells')
      .addItem('📊 Process Multiple Courses', 'processBulkCourses')
      .addSeparator()
      .addItem('📋 Setup Main Sheet', 'quickSetup')
      .addItem('📊 Setup Bulk Sheet', 'setupBulkTemplate')
      .addSeparator()
      .addItem('🔐 Test API Keys Library', 'testDataBridgeLibrary')
      .addItem('🔄 Reset Current Sheet', 'resetCourseraInputs')
      .addSeparator()
      .addSubMenu(ui.createMenu('📊 Email Generation Analytics')
        .addItem('📈 View Email Analytics (30 days)', 'showEmailGenerationAnalytics')
        .addItem('📋 Initialize Email Logging', 'initializeEmailLogger')
        .addItem('💾 Export Email Generation Logs', 'exportEmailGenerationLogs')
        .addItem('🔍 View Email Log Sheet', 'openEmailLogSheet'))
      .addSeparator()
      .addItem('📖 Help & Instructions', 'showHelp')
      .addToUi();
    
    Logger.log("Secure email generator menu with API keys library testing created!");
  }
  
  /**
   * Handle sheet edits (simplified - only email generation is logged)
   */
  function onEdit(e) {
    if (!e || !e.range) return;
    
    const range = e.range;
    const clickedCell = range.getA1Notation();
    const cellValue = range.getValue().toString();
    
    Logger.log(`ℹ️ Cell edited: ${clickedCell}, value: "${cellValue}". Use menu for email generation.`);
  }
  
  /**
   * Open email generation log sheet for viewing
   */
  function openEmailLogSheet() {
    try {
      const ss = SpreadsheetApp.getActiveSpreadsheet();
      let emailLogSheet = ss.getSheetByName("Email_Generation_Logs");
      
      if (!emailLogSheet) {
        // Initialize if doesn't exist
        if (typeof initializeEmailLogger === 'function') {
          emailLogSheet = initializeEmailLogger();
        } else {
          SpreadsheetApp.getActiveSpreadsheet().toast("❌ Email generation logging not initialized. Add email_generation_logger.js to your project.", "Error", 5);
          return;
        }
      }
      
      // Switch to the email log sheet
      ss.setActiveSheet(emailLogSheet);
      SpreadsheetApp.getActiveSpreadsheet().toast("📊 Opened email generation log sheet", "Email Generation Logs", 3);
      
    } catch (error) {
      Logger.log(`❌ Error opening email log sheet: ${error.message}`);
      SpreadsheetApp.getActiveSpreadsheet().toast(`❌ Error: ${error.message}`, "Error", 5);
    }
  }
  /**
   * =======================================
   * EMAIL GENERATION LOGGER - SIMPLIFIED
   * =======================================
   * 
   * Tracks only email generation activities:
   * - Single course email generation
   * - Bulk course email processing
   * - Processing performance and success rates
   */
  
  /**
   * ===============================
   * EMAIL LOGGING CONFIGURATION
   * ===============================
   */
  
  const EMAIL_LOG_CONFIG = {
    LOG_SHEET_NAME: "Email_Generation_Logs",
    MAX_LOG_ENTRIES: 10000,
    LOG_RETENTION_DAYS: 365 // 1 year of email generation logs
  };
  
  /**
   * ===============================
   * EMAIL LOG INITIALIZATION
   * ===============================
   */
  
  /**
   * Initialize email generation logging system
   */
  function initializeEmailLogger() {
    try {
      const ss = SpreadsheetApp.getActiveSpreadsheet();
      let logSheet = ss.getSheetByName(EMAIL_LOG_CONFIG.LOG_SHEET_NAME);
      
      if (!logSheet) {
        logSheet = ss.insertSheet(EMAIL_LOG_CONFIG.LOG_SHEET_NAME);
        
        // Create email generation log headers (10 fields)
        const headers = [
          "Timestamp",
          "User Email",
          "Action Details", 
          "Number of Courses",
          "Course-slug",
          "Email Tone",
          "Number of Emails generated",
          "Processing time (in minutes)",
          "Success or failures",
          "errorMessage"
        ];
        
        // Set headers
        logSheet.getRange(1, 1, 1, headers.length).setValues([headers]);
        
        // Style headers with professional blue theme
        const headerRange = logSheet.getRange(1, 1, 1, headers.length);
        headerRange.setBackground("#0056D3")
          .setFontColor("white")
          .setFontWeight("bold")
          .setHorizontalAlignment("center")
          .setVerticalAlignment("middle")
          .setBorder(true, true, true, true, false, false, "#003D99", SpreadsheetApp.BorderStyle.SOLID_MEDIUM);
        
        // Set column widths for readability
        logSheet.setColumnWidth(1, 150); // Timestamp
        logSheet.setColumnWidth(2, 200); // User Email
        logSheet.setColumnWidth(3, 200); // Action Details
        logSheet.setColumnWidth(4, 120); // Number of Courses
        logSheet.setColumnWidth(5, 180); // Course-slug
        logSheet.setColumnWidth(6, 120); // Email Tone
        logSheet.setColumnWidth(7, 120); // Number of Emails generated
        logSheet.setColumnWidth(8, 150); // Processing time (in minutes)
        logSheet.setColumnWidth(9, 120); // Success or failures
        logSheet.setColumnWidth(10, 300); // errorMessage
        
        // Freeze header row
        logSheet.setFrozenRows(1);
        
        Logger.log("✅ Email generation logging initialized successfully");
        SpreadsheetApp.getActiveSpreadsheet().toast("✅ Email generation logging initialized", "Setup Complete", 3);
      }
      
      return logSheet;
      
    } catch (error) {
      Logger.log(`❌ Error initializing email logger: ${error.message}`);
      throw error;
    }
  }
  
  /**
   * ===============================
   * EMAIL GENERATION LOGGING
   * ===============================
   */
  
  /**
   * Log email generation activity (single or bulk)
   */
  function logEmailGeneration(actionDetails, courseSlug, emailTone, emailCount, processingTimeMs, success, errorMessage = '') {
    try {
      const ss = SpreadsheetApp.getActiveSpreadsheet();
      let logSheet = ss.getSheetByName(EMAIL_LOG_CONFIG.LOG_SHEET_NAME);
      
      // Initialize if doesn't exist
      if (!logSheet) {
        logSheet = initializeEmailLogger();
      }
      
      // Get user information
      const userEmail = Session.getActiveUser().getEmail() || 'Unknown User';
      const timestamp = new Date();
      const processingTimeMinutes = Math.round(processingTimeMs / 60000 * 100) / 100; // Convert to minutes and round to 2 decimal places
      const successStatus = success ? 'Success' : 'Failure';
      
      // Determine number of courses (for single vs bulk operations)
      const numberOfCourses = actionDetails.includes('Bulk') ? 
        (courseSlug.includes(',') ? courseSlug.split(',').length : 1) : 1;
      
      // Prepare log entry with 10 fields
      const logEntry = [
        timestamp,                                    // 1. Timestamp
        userEmail,                                   // 2. User Email
        actionDetails,                               // 3. Action Details
        numberOfCourses,                             // 4. Number of Courses
        courseSlug || 'N/A',                        // 5. Course-slug
        emailTone || 'Professional',                // 6. Email Tone
        emailCount || 0,                            // 7. Number of Emails generated
        processingTimeMinutes,                      // 8. Processing time (in minutes)
        successStatus,                              // 9. Success or failures
        errorMessage || ''                          // 10. errorMessage
      ];
      
      // Add to sheet
      const nextRow = logSheet.getLastRow() + 1;
      logSheet.getRange(nextRow, 1, 1, logEntry.length).setValues([logEntry]);
      
      // Style the row based on success/failure
      const rowRange = logSheet.getRange(nextRow, 1, 1, logEntry.length);
      if (success) {
        rowRange.setBackground("#F8F9FA"); // Light gray for success
      } else {
        rowRange.setBackground("#FFEBEE"); // Light red for failure
        logSheet.getRange(nextRow, 8).setFontColor("#C62828"); // Red text for failure status
      }
      
      // Set borders and alignment
      rowRange.setBorder(false, false, true, false, false, false, "#DEE2E6", SpreadsheetApp.BorderStyle.SOLID);
      rowRange.setVerticalAlignment("middle");
      
      // Format timestamp column
      logSheet.getRange(nextRow, 1).setNumberFormat("yyyy-mm-dd hh:mm:ss");
      
      // Center align numeric columns
      logSheet.getRange(nextRow, 4).setHorizontalAlignment("center"); // Number of Courses
      logSheet.getRange(nextRow, 7).setHorizontalAlignment("center"); // Number of Emails generated
      logSheet.getRange(nextRow, 8).setHorizontalAlignment("center"); // Processing time (in minutes)
      logSheet.getRange(nextRow, 9).setHorizontalAlignment("center"); // Success or failures
      
      Logger.log(`📊 Email generation logged: ${actionDetails} - ${successStatus}`);
      
      // Clean up old entries if needed
      cleanupOldEmailLogs();
      
    } catch (error) {
      Logger.log(`❌ Error logging email generation: ${error.message}`);
      // Don't throw error to avoid disrupting email generation process
    }
  }
  
  /**
   * Log single course email generation
   */
  function logSingleEmailGeneration(courseSlug, emailTone, emailCount, processingTimeMs, success, errorMessage = '') {
    const actionDetails = `Single Course Email Generation`;
    logEmailGeneration(actionDetails, courseSlug, emailTone, emailCount, processingTimeMs, success, errorMessage);
  }
  
  /**
   * Log bulk course email processing
   */
  function logBulkEmailGeneration(courseSlugs, emailTone, totalEmails, processingTimeMs, successCount, failureCount, errorMessage = '') {
    const actionDetails = `Bulk Email Generation (${courseSlugs.length} courses)`;
    const courseSlugList = courseSlugs.join(', ');
    const success = failureCount === 0;
    const finalErrorMessage = failureCount > 0 ? `${failureCount} failures: ${errorMessage}` : errorMessage;
    
    logEmailGeneration(actionDetails, courseSlugList, emailTone, totalEmails, processingTimeMs, success, finalErrorMessage);
  }
  
  /**
   * ===============================
   * EMAIL LOG MANAGEMENT
   * ===============================
   */
  
  /**
   * Clean up old email generation logs
   */
  function cleanupOldEmailLogs() {
    try {
      const ss = SpreadsheetApp.getActiveSpreadsheet();
      const logSheet = ss.getSheetByName(EMAIL_LOG_CONFIG.LOG_SHEET_NAME);
      
      if (!logSheet) return;
      
      const lastRow = logSheet.getLastRow();
      if (lastRow <= EMAIL_LOG_CONFIG.MAX_LOG_ENTRIES) return;
      
      // Calculate how many rows to delete
      const rowsToDelete = lastRow - EMAIL_LOG_CONFIG.MAX_LOG_ENTRIES;
      
      // Delete oldest entries (keeping header row)
      logSheet.deleteRows(2, rowsToDelete);
      
      Logger.log(`🧹 Cleaned up ${rowsToDelete} old email generation log entries`);
      
    } catch (error) {
      Logger.log(`❌ Error cleaning up email logs: ${error.message}`);
    }
  }
  
  /**
   * ===============================
   * EMAIL LOG ANALYTICS
   * ===============================
   */
  
  /**
   * Get email generation analytics
   */
  function getEmailGenerationAnalytics(days = 30) {
    try {
      const ss = SpreadsheetApp.getActiveSpreadsheet();
      const logSheet = ss.getSheetByName(EMAIL_LOG_CONFIG.LOG_SHEET_NAME);
      
      if (!logSheet) {
        return "No email generation logs found. Please initialize the logger first.";
      }
      
      const lastRow = logSheet.getLastRow();
      if (lastRow <= 1) {
        return "No email generation data available.";
      }
      
      // Get all data (10 columns now)
      const data = logSheet.getRange(2, 1, lastRow - 1, 10).getValues();
      const cutoffDate = new Date();
      cutoffDate.setDate(cutoffDate.getDate() - days);
      
      // Filter recent data
      const recentData = data.filter(row => new Date(row[0]) >= cutoffDate);
      
      if (recentData.length === 0) {
        return `No email generation activity in the last ${days} days.`;
      }
      
      // Calculate analytics (updated for new 10-field format)
      const totalGenerations = recentData.length;
      const successfulGenerations = recentData.filter(row => row[8] === 'Success').length; // Success/failures (column 8)
      const failedGenerations = totalGenerations - successfulGenerations;
      const successRate = Math.round((successfulGenerations / totalGenerations) * 100);
      
      const totalEmailsGenerated = recentData.reduce((sum, row) => sum + (row[6] || 0), 0); // Number of Emails generated (column 6)
      const totalCourses = recentData.reduce((sum, row) => sum + (row[3] || 0), 0); // Number of Courses (column 3)
      const averageProcessingTime = recentData.reduce((sum, row) => sum + (row[7] || 0), 0) / totalGenerations; // Processing time in minutes (column 7)
      
      // Single vs Bulk operations
      const singleGenerations = recentData.filter(row => row[2].includes('Single Course')).length; // Action Details (column 2)
      const bulkGenerations = recentData.filter(row => row[2].includes('Bulk')).length; // Action Details (column 2)
      
      // Top courses
      const courseCount = {};
      recentData.forEach(row => {
        const course = row[4]; // Course-slug (column 4)
        if (course && course !== 'N/A') {
          courseCount[course] = (courseCount[course] || 0) + 1;
        }
      });
      
      const topCourses = Object.entries(courseCount)
        .sort(([,a], [,b]) => b - a)
        .slice(0, 5)
        .map(([course, count]) => `${course}: ${count} generations`);
      
      // Tone usage
      const toneCount = {};
      recentData.forEach(row => {
        const tone = row[5] || 'Professional'; // Email Tone (column 5)
        toneCount[tone] = (toneCount[tone] || 0) + 1;
      });
      
      const toneUsage = Object.entries(toneCount)
        .map(([tone, count]) => `${tone}: ${count} (${Math.round(count/totalGenerations*100)}%)`)
        .join(', ');
      
      const analytics = `📊 EMAIL GENERATION ANALYTICS (Last ${days} days)
  
  📈 OVERVIEW:
  • Total Generations: ${totalGenerations}
  • Single Course: ${singleGenerations}
  • Bulk Processing: ${bulkGenerations}
  • Total Courses Processed: ${totalCourses}
  • Total Emails Generated: ${totalEmailsGenerated}
  • Success Rate: ${successRate}%
  
  🎯 TOP COURSES:
  ${topCourses.length > 0 ? topCourses.join('\n') : 'No course data available'}
  
  🎨 TONE USAGE:
  ${toneUsage}
  
  📊 PERFORMANCE:
  • Successful Generations: ${successfulGenerations}
  • Failed Generations: ${failedGenerations}
  • Success Rate: ${successRate}%
  • Average Processing Time: ${averageProcessingTime.toFixed(2)} minutes per generation
  • Total Email Volume: ${totalEmailsGenerated} emails`;
      
      Logger.log(analytics);
      return analytics;
      
    } catch (error) {
      Logger.log(`❌ Error getting email analytics: ${error.message}`);
      return `Error retrieving analytics: ${error.message}`;
    }
  }
  
  /**
   * Show email generation analytics in UI
   */
  function showEmailGenerationAnalytics() {
    try {
      const analytics = getEmailGenerationAnalytics(30);
      
      // Show in UI alert
      const ui = SpreadsheetApp.getUi();
      ui.alert('📊 Email Generation Analytics', analytics, ui.ButtonSet.OK);
      
    } catch (error) {
      Logger.log(`❌ Error showing email analytics: ${error.message}`);
      SpreadsheetApp.getActiveSpreadsheet().toast(`❌ Error: ${error.message}`, "Analytics Error", 5);
    }
  }
  
  /**
   * Export email generation logs to CSV
   */
  function exportEmailGenerationLogs() {
    try {
      const ss = SpreadsheetApp.getActiveSpreadsheet();
      const logSheet = ss.getSheetByName(EMAIL_LOG_CONFIG.LOG_SHEET_NAME);
      
      if (!logSheet) {
        SpreadsheetApp.getActiveSpreadsheet().toast("❌ No email generation logs found", "Export Error", 5);
        return;
      }
      
      const lastRow = logSheet.getLastRow();
      if (lastRow <= 1) {
        SpreadsheetApp.getActiveSpreadsheet().toast("❌ No email generation data to export", "Export Error", 5);
        return;
      }
      
      // Get all data including headers (10 columns now)
      const data = logSheet.getRange(1, 1, lastRow, 10).getValues();
      
      // Convert to CSV format
      const csvContent = data.map(row => 
        row.map(cell => {
          if (cell instanceof Date) {
            return cell.toISOString();
          }
          // Escape quotes and wrap in quotes if contains comma
          const cellStr = String(cell);
          if (cellStr.includes(',') || cellStr.includes('"') || cellStr.includes('\n')) {
            return '"' + cellStr.replace(/"/g, '""') + '"';
          }
          return cellStr;
        }).join(',')
      ).join('\n');
      
      // Create blob and download
      const blob = Utilities.newBlob(csvContent, 'text/csv', `email_generation_logs_${new Date().toISOString().split('T')[0]}.csv`);
      
      SpreadsheetApp.getActiveSpreadsheet().toast("✅ Email generation logs prepared for export", "Export Complete", 5);
      Logger.log("📊 Email generation logs exported successfully");
      
      return blob;
      
    } catch (error) {
      Logger.log(`❌ Error exporting email logs: ${error.message}`);
      SpreadsheetApp.getActiveSpreadsheet().toast(`❌ Export error: ${error.message}`, "Export Error", 5);
    }
  }
  
  /**
   * Open email generation log sheet
   */
  function openEmailLogSheet() {
    try {
      const ss = SpreadsheetApp.getActiveSpreadsheet();
      let logSheet = ss.getSheetByName(EMAIL_LOG_CONFIG.LOG_SHEET_NAME);
      
      if (!logSheet) {
        logSheet = initializeEmailLogger();
      }
      
      ss.setActiveSheet(logSheet);
      SpreadsheetApp.getActiveSpreadsheet().toast("📊 Opened email generation log sheet", "Email Logs", 3);
      
    } catch (error) {
      Logger.log(`❌ Error opening email log sheet: ${error.message}`);
      SpreadsheetApp.getActiveSpreadsheet().toast(`❌ Error: ${error.message}`, "Error", 5);
    }
  }
  
  