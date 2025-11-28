/**
 * SECURE API KEYS PROJECT
 * This project stores API keys in Script Properties and contains all sensitive business logic
 * Deploy as library - only you have access
 * 
 * Contains:
 * - API key management
 * - Databricks data querying
 * - Gemini AI integration
 * - Email prompt generation
 */

function setupApiKeys() {
    // REPLACE THESE WITH YOUR ACTUAL API KEYS
    const properties = PropertiesService.getScriptProperties();
    
    Logger.log('✅ API keys have been set up securely');
    return 'API keys configured successfully!';
  }
  
  // Functions that your main sheet will call
  function getGeminiApiKey() {
    return PropertiesService.getScriptProperties().getProperty('GEMINI_API_KEY');
  }
  
  function getDbAccessToken() {
    return PropertiesService.getScriptProperties().getProperty('DB_ACCESS_TOKEN');
  }
  
  function getWarehouseId() {
    return PropertiesService.getScriptProperties().getProperty('warehouseId');
  }
  
  function getServer() {
    return PropertiesService.getScriptProperties().getProperty('server');
  }
  
  // Convenient function to get all keys at once
  function getAllApiKeys() {
    const properties = PropertiesService.getScriptProperties();
    return {
      GEMINI_API_KEY: properties.getProperty('GEMINI_API_KEY'),
      DB_ACCESS_TOKEN: properties.getProperty('DB_ACCESS_TOKEN'),
      warehouseId: properties.getProperty('warehouseId'),
      server: properties.getProperty('server')
    };
  }
  
  // Test function to verify keys are stored
  function testApiKeys() {
    const keys = getAllApiKeys();
    Logger.log('🔍 API Keys Status:');
    Logger.log('Gemini Key: ' + (keys.GEMINI_API_KEY ? '✅ Set (' + keys.GEMINI_API_KEY.substring(0, 10) + '...)' : '❌ Missing'));
    Logger.log('DB Token: ' + (keys.DB_ACCESS_TOKEN ? '✅ Set (' + keys.DB_ACCESS_TOKEN.substring(0, 10) + '...)' : '❌ Missing'));
    Logger.log('Warehouse ID: ' + (keys.warehouseId ? '✅ Set (' + keys.warehouseId + ')' : '❌ Missing'));
    Logger.log('Server: ' + (keys.server ? '✅ Set (' + keys.server + ')' : '❌ Missing'));
    return 'Check execution log for API keys status';
  }
  
  /**
   * ===============================
   * GEMINI AI INTEGRATION
   * ===============================
   */
  
  /**
   * Call Gemini AI API to generate content
   */
  function callGemini(promptText) {
    const GEMINI_API_KEY = getGeminiApiKey();
    
    if (!GEMINI_API_KEY) {
      throw new Error("Gemini API key not available. Please check your DataBridge configuration.");
    }
    
    const model = "gemini-2.5-flash";
    const url = `https://generativelanguage.googleapis.com/v1/models/${model}:generateContent?key=${GEMINI_API_KEY}`;
  
    const payload = {
      contents: [
        {
          parts: [
            {
              text: promptText
            }
          ]
        }
      ],
      generationConfig: {
        temperature: 0.7,
        maxOutputTokens: 4096
      }
    };
  
    const options = {
      method: "post",
      contentType: "application/json",
      payload: JSON.stringify(payload)
    };
  
    try {
      const response = UrlFetchApp.fetch(url, options);
      const result = JSON.parse(response.getContentText());
      
      if (!result.candidates || !result.candidates[0]) {
        throw new Error("No candidates returned from Gemini API");
      }
      
      const candidate = result.candidates[0];
      
      // Check for errors
      if (candidate.finishReason === "MAX_TOKENS") {
        throw new Error("Response truncated due to token limit");
      }
      
      if (candidate.finishReason && candidate.finishReason !== "STOP") {
        throw new Error(`Response generation stopped: ${candidate.finishReason}`);
      }
      
      // Return content
      if (candidate.content && candidate.content.parts && candidate.content.parts[0] && candidate.content.parts[0].text) {
        return candidate.content.parts[0].text;
      }
      
      throw new Error(`Unknown response structure: ${JSON.stringify(candidate)}`);
      
    } catch (error) {
      console.error("Error calling Gemini API:", error);
      throw new Error(`Failed to generate content: ${error.message}`);
    }
  }
  
  /**
   * ===============================
   * DATABRICKS DATA INTEGRATION
   * ===============================
   */
  
  /**
   * Get course data from Databricks with full details
   */
  function getCourseDataFull(courseSlug) {
    Logger.log(`🔍 getCourseDataFull for: ${courseSlug}`);
    
    // Validate input
    if (!courseSlug || typeof courseSlug !== 'string') {
      throw new Error("Course slug is required and must be a string");
    }
  
    const accessToken = getDbAccessToken();
    const warehouseId = getWarehouseId();
    const server = getServer();
  
    // Verify configuration
    if (!accessToken || !warehouseId || !server) {
      throw new Error("Databricks configuration incomplete. Please check your DataBridge configuration for DB_ACCESS_TOKEN, warehouseId, and server.");
    }
  
    const query = `
  WITH live_course_branches AS (
      SELECT
          t.course_id,
          t.course_branch_id AS latest_live_course_branch_id
      FROM (
          SELECT
              course_id,
              course_branch_id,
              ROW_NUMBER() OVER (
                  PARTITION BY course_id
                  ORDER BY course_branch_launched_ts DESC NULLS LAST
              ) AS rn
          FROM prod.gold.course_branches_vw
          -- WHERE
          --     course_branch_status != 'Archived'
          --     AND is_course_branch_public = TRUE
      ) t
      WHERE t.rn = 1
  ),
  
  module_details AS (
      SELECT
          course_id,
          course_branch_id,
          CONCAT_WS('\\n|| ', TRANSFORM(sorted_array, x -> x.course_branch_module_order)) AS course_branch_module_order,
          CONCAT_WS('\\n|| ', TRANSFORM(sorted_array, x -> x.course_branch_module_name)) AS course_branch_module_name,
          CONCAT_WS('\\n|| ', TRANSFORM(sorted_array, x -> x.course_branch_module_desc)) AS course_branch_module_desc
      FROM (
          SELECT
              course_id,
              course_branch_id,
              SORT_ARRAY(
                  COLLECT_LIST(
                      NAMED_STRUCT(
                          'course_branch_module_order', course_branch_module_order + 1,
                          'course_branch_module_name', course_branch_module_name,
                          'course_branch_module_desc', course_branch_module_desc
                      )
                  )
              ) AS sorted_array
          FROM prod.gold.course_branch_modules_vw
          GROUP BY course_id, course_branch_id
      ) s
  )
  
  SELECT
      c.course_name,
      c.course_id,
      c.course_workload,
      c.course_slug,
      d.course_url,
      c.course_type,
      c.learning_objectives,
      d.all_skills,
      d.partner_names,
      c.course_desc,
      c.course_language_name,
      m.course_branch_module_name,
      m.course_branch_module_desc,
      m.course_branch_module_order
  FROM prod.gold.courses_vw c
  JOIN live_course_branches lcb
    ON c.course_id = lcb.course_id
  JOIN module_details m
    ON lcb.latest_live_course_branch_id = m.course_branch_id
  JOIN coursera_warehouse.edw_bi.enterprise_catalog_course_metadata d
    ON c.course_id = d.course_id
  WHERE
      c.is_test_course = FALSE
      AND c.course_is_limited_to_enterprise = FALSE
      AND c.course_slug = '${courseSlug}'
  ORDER BY c.course_name
  `;
  
    const url = `${server}/api/2.0/sql/statements`;
    const payload = { statement: query, warehouse_id: warehouseId };
  
    const options = {
      method: "POST",
      headers: {
        "Authorization": "Bearer " + accessToken,
        "Content-Type": "application/json"
      },
      payload: JSON.stringify(payload)
    };
  
    const response = UrlFetchApp.fetch(url, options);
    const result = JSON.parse(response.getContentText());
    
    const statementId = result.statement_id;
    const fetchUrl = `${server}/api/2.0/sql/statements/${statementId}`;
  
    // Poll for results
    let fetchData, data;
    for (let i = 0; i < 50; i++) {
      Utilities.sleep(6000);
      
      const fetchResponse = UrlFetchApp.fetch(fetchUrl, {
        method: "GET",
        headers: { "Authorization": "Bearer " + accessToken }
      });
      fetchData = JSON.parse(fetchResponse.getContentText());
      
      if (fetchData.status?.state === "SUCCEEDED" && fetchData.result?.data_array) {
        data = fetchData.result.data_array;
        break;
      } else if (fetchData.status?.state === "FAILED") {
        Logger.log("❌ Query FAILED!");
        Logger.log("Error details:", JSON.stringify(fetchData, null, 2));
        break;
      }
    }
  
    // Check results
    if (!data || data.length === 0) {
      if (fetchData.status?.state === "SUCCEEDED") {
        throw new Error(`Course with slug '${courseSlug}' returned no data from Databricks (query succeeded but empty result)`);
      } else {
        throw new Error(`Databricks query failed with state: ${fetchData.status?.state || 'unknown'}`);
      }
    }
  
    // Process the data
    const row = data[0];
    
    // Helper function to normalize course URLs
    function normalizeUrl(url) {
      if (!url) return url;
      return url.startsWith('http') ? url : `https://www.coursera.org/learn/${url}`;
    }
  
    // Helper function to create modules JSON
    function createModulesJson(courseData) {
      try {
        const modules = [];
        
        // Debug logging
        Logger.log('🔍 Module data received:');
        Logger.log('Module names raw: ' + JSON.stringify(courseData.course_branch_module_name));
        Logger.log('Module descriptions raw: ' + JSON.stringify(courseData.course_branch_module_desc));
        
        if (courseData.course_branch_module_name && courseData.course_branch_module_desc) {
          const names = courseData.course_branch_module_name.split('\n|| ');
          const descriptions = courseData.course_branch_module_desc.split('\n|| ');
          
          Logger.log('📊 Parsed modules:');
          Logger.log('Names array length: ' + names.length);
          Logger.log('Names: ' + JSON.stringify(names));
          Logger.log('Descriptions array length: ' + descriptions.length);
          
          for (let i = 0; i < names.length; i++) {
            if (names[i] && names[i].trim()) {
              modules.push({
                name: names[i].trim(),
                description: (descriptions[i] && descriptions[i].trim()) ? descriptions[i].trim() : names[i].trim()
              });
            }
          }
        }
        
        Logger.log('✅ Final modules created: ' + modules.length + ' modules');
        Logger.log('Modules: ' + JSON.stringify(modules));
        
        return modules;
      } catch (error) {
        Logger.log(`⚠️ Error creating modules JSON: ${error.message}`);
        return [];
      }
    }
  
    const course = {
      course_name: row[0],
      course_id: row[1], 
      course_workload: row[2],
      course_slug: row[3],
      course_url: normalizeUrl(row[4]),
      course_type: row[5],
      learning_objectives: row[6],
      all_skills: row[7] ? (typeof row[7] === 'string' ? row[7].split(',').map(s => s.trim()) : row[7]) : [],
      partner_names: row[8],
      course_desc: row[9],
      course_language_name: row[10],
      course_branch_module_name: row[11],
      course_branch_module_desc: row[12],
      course_branch_module_order: row[13],
      instructor_names: "Course Instructor"
    };
  
    // Create modules_json from module data
    course.modules_json = createModulesJson(course);
  
    Logger.log("✅ getCourseDataFull completed successfully!");
    return course;
  }
  
  /**
   * ===============================
   * EMAIL PROMPT GENERATION
   * ===============================
   */
  
  /**
   * Generate all email prompts for a course
   */
  function generateEmailPrompts(course, userName, tone = 'Professional') {
    const courseInfo = {
      name: course.course_name,
      desc: course.course_desc,
      skills: course.all_skills,
      modules: course.modules_json,
      partner: course.partner_names || "Course Partner"
    };
    
    // Debug logging
    Logger.log('🎯 generateEmailPrompts called:');
    Logger.log('Course name: ' + courseInfo.name);
    Logger.log('Modules received: ' + JSON.stringify(courseInfo.modules));
    Logger.log('Module count: ' + (courseInfo.modules ? courseInfo.modules.length : 0));
    
    // Consistent signature for all emails
    const emailSignature = `Happy Learning,\n${courseInfo.partner}`;
  
    // Tone-specific styling instructions
    const toneInstructions = {
      'Professional': 'Use conversational yet professional language. Be friendly, clear, and approachable. Avoid overly formal phrases like "We are pleased" - use natural, warm language instead.',
      'Motivational': 'Use encouraging, supportive language with a conversational tone. Be inspiring but balanced - avoid being overly energetic or overwhelming. Focus on achievement and growth with natural, genuine phrasing.',
      'Descriptive': 'Use detailed, informative language in a conversational style. Provide comprehensive explanations naturally. Focus on thorough understanding with friendly, approachable tone.'
    };
  
    const selectedToneStyle = toneInstructions[tone] || toneInstructions['Professional'];
  
    const prompts = [];
  
    // Welcome Email
    const welcomePrompt = `Create a welcome email for ${userName} starting ${courseInfo.name} in a ${tone.toLowerCase()} tone.
  
  Course: ${courseInfo.name}
  Key Skills: ${courseInfo.skills.slice(0, 4).join(', ')}
  Tone Style: ${selectedToneStyle}
  
  Format: Concise, engaging email with exactly 3 paragraphs (4-5 lines each):
  DO NOT include a subject line - start directly with the email content.
  
  PARAGRAPH 1 (4-5 lines): Begin with "Hi ${userName}," followed by a blank line, then warm welcome to the course
  PARAGRAPH 2 (4-5 lines): Brief overview of what they'll learn and key skills they'll develop  
  PARAGRAPH 3 (4-5 lines): Encouraging message about their learning journey and what to expect
  
  IMPORTANT FORMATTING RULES:
  - NEVER use asterisks (*) to highlight words like *vocabulary*, *grammar*, *skills*, etc.
  - NEVER put course names in quotation marks
  - DO NOT mention specific module names or module structure
  - Add only ONE subtle emoji for warmth (🚀 for motivation OR 🌟 for achievement)
  - Keep it concise - exactly 3 paragraphs with 4-5 lines each
  - Include a blank line after "Hi ${userName}," before starting the main content
  
  Tone: Conversational and warm - like talking to a friend who's starting something exciting. Avoid formal phrases like "We are pleased" or "We wish to inform". Use natural, friendly language with only one subtle emoji for warmth.
  Length: Concise 3 paragraphs, each with 4-5 lines. Prioritize engagement over length.
  End: "${emailSignature}"`;
    prompts.push({ type: "Welcome to the course", prompt: welcomePrompt });
  
    // Module Emails
    if (courseInfo.modules && courseInfo.modules.length > 0) {
      Logger.log('📚 Processing ' + courseInfo.modules.length + ' modules for email generation');
      courseInfo.modules.forEach((mod, idx) => {
        Logger.log(`Processing module ${idx + 1}: ${mod.name}`);
        const isFirstModule = idx === 0;
        const isLastModule = idx === courseInfo.modules.length - 1;
        
        const modulePrompt = `Create an encouraging email for ${userName} about Module ${idx + 1}: ${mod.name} of ${courseInfo.name} in a ${tone.toLowerCase()} tone.
  
  Context: Module ${idx + 1} of ${courseInfo.modules.length}
  ${isFirstModule ? 'This is their first module - help them get started successfully.' : ''}
  ${isLastModule ? 'This is the final module - encourage them to finish strong.' : ''}
  Tone Style: ${selectedToneStyle}
  
  Format: Concise, motivational email with exactly 3 paragraphs (4-5 lines each):
  DO NOT include a subject line - start directly with the email content.
  
  PARAGRAPH 1 (4-5 lines): Begin with "Hi ${userName}," followed by a blank line, then ${isFirstModule ? 'encouraging start for their first module' : isLastModule ? 'motivating message for the final push' : 'acknowledgment of their progress so far'}
  PARAGRAPH 2 (4-5 lines): ${isFirstModule ? 'Practical tips for starting strong (goal-setting, study schedule, momentum)' : 
       isLastModule ? 'Encouragement to finish strong and reflection on their journey' : 
       'Study advice and momentum maintenance with motivation'}
  PARAGRAPH 3 (4-5 lines): Module-specific encouragement and positive reinforcement about their learning journey
  
  IMPORTANT FORMATTING RULES:
  - NEVER use asterisks (*) to highlight words like *vocabulary*, *grammar*, *skills*, etc.
  - NEVER put course names or module names in quotation marks
  - Add only ONE subtle emoji for motivation (🚀 OR 💪 OR ⭐) to make the tone warmer without overwhelming
  - Keep it concise - exactly 3 paragraphs with 4-5 lines each
  - Include a blank line after "Hi ${userName}," before starting the main content
  
  Tone: Conversational and ${tone.toLowerCase()} - like a supportive friend checking in. Be encouraging but not overly energetic. Avoid formal phrases like "We are pleased that you begin" - use natural, friendly language with one subtle emoji. 
  Length: Concise 3 paragraphs, each with 4-5 lines. Prioritize engagement and motivation over length.
  End: "${emailSignature}"`;
        prompts.push({ type: `Get ready for Module ${idx + 1}`, prompt: modulePrompt });
        Logger.log(`✅ Added prompt for module ${idx + 1}`);
      });
    } else {
      Logger.log('⚠️ No modules found or modules array is empty');
    }
  
    // Announcement Emails (3 retention emails)
    const announcement1Prompt = `Create a retention announcement email for ${userName} highlighting the career value and professional benefits of ${courseInfo.name} in a ${tone.toLowerCase()} tone.
  
  Course: ${courseInfo.name}
  Course Description: ${courseInfo.desc || 'Professional skill development course'}
  Key Skills: ${courseInfo.skills.slice(0, 4).join(', ')}
  Tone Style: ${selectedToneStyle}
  
  Format: Concise retention email with exactly 2 paragraphs (4-5 lines each):
  DO NOT include a subject line - start directly with the email content.
  
  PARAGRAPH 1 (4-5 lines): Begin with "Hi ${userName}," followed by a blank line, then highlight the valuable career opportunities and market demand for these specific skills
  PARAGRAPH 2 (4-5 lines): Focus on concrete benefits - salary ranges, job market growth, industry demand - and encourage them to continue building these in-demand skills
  
  IMPORTANT FORMATTING RULES:
  - NEVER use asterisks (*) to highlight words like *vocabulary*, *grammar*, *skills*, etc.
  - NEVER put course names in quotation marks
  - NEVER mention fictional names or made-up stories - stick to factual market data and real industry trends
  - NEVER use phrases like "Through this project" or project-specific language
  - Add only ONE motivating emoji for career growth (🚀 OR 📈 OR ⭐) to create excitement without overwhelming
  - Keep it concise - exactly 2 paragraphs with 4-5 lines each
  - Include a blank line after "Hi ${userName}," before starting the main content
  
  Tone: Conversational and ${tone.toLowerCase()} - like a career mentor highlighting opportunities. Focus on career benefits, skill value, and professional growth incentives.
  ${tone === 'Professional' ? 'Focus on: Career advancement opportunities, market value of skills, professional credibility and advancement' : 
    tone === 'Motivational' ? 'Focus on: Achieving career dreams, unlocking potential, transforming professional life through skill development' : 
    'Focus on: Comprehensive skill mastery, detailed career applications, thorough professional development benefits'}
  Length: Concise 2 paragraphs, each with 4-5 lines. Prioritize factual career value and real market incentives.
  End: "${emailSignature}"`;
  
    const announcement2Prompt = `Create a retention announcement email for ${userName} focusing on building learning momentum and maintaining consistent progress in ${courseInfo.name} in a ${tone.toLowerCase()} tone.
  
  Course: ${courseInfo.name}
  Course Description: ${courseInfo.desc || 'Professional skill development course'}
  Key Skills: ${courseInfo.skills.slice(0, 4).join(', ')}
  Tone Style: ${selectedToneStyle}
  
  Format: Concise momentum-focused retention email with exactly 2 paragraphs (4-5 lines each):
  DO NOT include a subject line - start directly with the email content.
  
  PARAGRAPH 1 (4-5 lines): Begin with "Hi ${userName}," followed by a blank line, then emphasize the power of consistent learning and how small daily progress compounds into major skill development
  PARAGRAPH 2 (4-5 lines): Provide actionable momentum tips like setting 15-minute daily learning blocks, and encourage them to continue building these valuable skills today
  
  IMPORTANT FORMATTING RULES:
  - NEVER use asterisks (*) to highlight words like *vocabulary*, *grammar*, *skills*, etc.
  - NEVER put course names in quotation marks
  - NEVER mention fictional names, success stories, or made-up examples - focus on factual learning principles
  - NEVER use phrases like "Through this project" or project-specific language
  - Add only ONE energizing emoji for momentum (💪 OR ⚡ OR 🎯) to create motivation without overwhelming
  - Keep it concise - exactly 2 paragraphs with 4-5 lines each
  - Include a blank line after "Hi ${userName}," before starting the main content
  
  Tone: Conversational and ${tone.toLowerCase()} - like an inspiring coach emphasizing the power of consistency. Focus on momentum building, habit formation, and the compound effect of regular learning.
  ${tone === 'Professional' ? 'Focus on: Professional growth through consistent skill building, career advancement through disciplined learning' : 
    tone === 'Motivational' ? 'Focus on: Building unstoppable momentum, creating powerful learning habits, achieving breakthroughs through consistency' : 
    'Focus on: Systematic skill development, comprehensive progress through regular practice, thorough mastery via consistent effort'}
  Length: Concise 2 paragraphs, each with 4-5 lines. Prioritize factual momentum building and practical learning habits.
  End: "${emailSignature}"`;
  
    const announcement3Prompt = `Create a retention announcement email for ${userName} showcasing real-world impact and success stories from ${courseInfo.name} to inspire continued learning in a ${tone.toLowerCase()} tone.
  
  Course: ${courseInfo.name}
  Course Description: ${courseInfo.desc || 'Professional skill development course'}
  Key Skills: ${courseInfo.skills.slice(0, 4).join(', ')}
  Tone Style: ${selectedToneStyle}
  
  Format: Concise impact-focused retention email with exactly 2 paragraphs (4-5 lines each):
  DO NOT include a subject line - start directly with the email content.
  
  PARAGRAPH 1 (4-5 lines): Begin with "Hi ${userName}," followed by a blank line, then highlight the real-world applications and industry demand for these specific skills
  PARAGRAPH 2 (4-5 lines): Focus on concrete opportunities - job roles, industry growth, practical applications - and encourage them to continue developing these market-relevant skills
  
  IMPORTANT FORMATTING RULES:
  - NEVER use asterisks (*) to highlight words like *vocabulary*, *grammar*, *skills*, etc.
  - NEVER put course names in quotation marks
  - NEVER mention fictional names, made-up success stories, or individual examples - stick to factual industry data
  - NEVER use phrases like "Through this project" or project-specific language
  - Add only ONE inspiring emoji for success (🌟 OR 🚀 OR 💫) to create aspiration without overwhelming
  - Keep it concise - exactly 2 paragraphs with 4-5 lines each
  - Include a blank line after "Hi ${userName}," before starting the main content
  
  Tone: Conversational and ${tone.toLowerCase()} - like an informed mentor highlighting market realities. Focus on real-world impact, tangible benefits, and factual industry applications.
  ${tone === 'Professional' ? 'Focus on: Industry demand, concrete career paths, measurable market value of skills' : 
    tone === 'Motivational' ? 'Focus on: Real opportunities, achievable career goals, practical skill applications' : 
    'Focus on: Comprehensive market analysis, detailed industry applications, thorough demonstration of skill relevance'}
  Length: Concise 2 paragraphs, each with 4-5 lines. Prioritize factual market impact and real industry applications.
  End: "${emailSignature}"`;
  
    // Completion Email
    const completionPrompt = `Create a completion email for ${userName} who just finished ${courseInfo.name} in a ${tone.toLowerCase()} tone.
  
  Course: ${courseInfo.name}
  Skills Gained: ${courseInfo.skills.slice(0, 6).join(', ')}
  Tone Style: ${selectedToneStyle}
  
  Write a genuine completion message with:
  Concise, celebratory email with exactly 3 paragraphs (4-5 lines each):
  DO NOT include a subject line - start directly with the email content.
  
  PARAGRAPH 1 (4-5 lines): Begin with "Hi ${userName}," followed by a blank line, then warm congratulations with one celebration emoji
  PARAGRAPH 2 (4-5 lines): Brief mention of skills gained and acknowledge their dedication with motivational content
  PARAGRAPH 3 (4-5 lines): "Here's how you can keep your momentum going:" with 2-3 concise actionable bullet points:
  • Showcase your accomplishment (LinkedIn sharing, certificate display)
  • Keep building (next learning steps, related courses)
  • Apply your skills (practical career applications)
  
  IMPORTANT FORMATTING RULES:
  - NEVER use asterisks (*) to highlight words like *vocabulary*, *grammar*, *skills*, etc.
  - NEVER put course names in quotation marks
  - Add only ONE congratulatory emoji for celebration (🎉 OR 🏆 OR 🌟) to make it warmer without overwhelming
  - Keep it concise - exactly 3 paragraphs with 4-5 lines each
  - Use bullet points (•) for actionable advice in paragraph 3
  - Include a blank line after "Hi ${userName}," before starting the main content
  
  Tone: Conversational and ${tone.toLowerCase()} - like a supportive mentor celebrating their achievement naturally. Avoid formal corporate language. Use warm, genuine, friendly phrasing with one celebratory emoji for warmth.
  ${tone === 'Professional' ? 'Avoid: Corporate speak, excessive praise, fake enthusiasm, formal phrases like "We are pleased"' : 
    tone === 'Motivational' ? 'Focus on: Achievement and growth with encouraging but balanced language - motivational without being overly energetic' : 
    'Focus on: Comprehensive acknowledgment with detailed but conversational language and one subtle emoji'}
  Length: Concise 3 paragraphs, each with 4-5 lines. Prioritize engagement and celebration over length.
  End: "${emailSignature}"`;
  
    // Add all prompts in the order they should be generated
    prompts.push({ type: "Congratulations", prompt: completionPrompt });
    prompts.push({ type: "Announcement 1", prompt: announcement1Prompt });
    prompts.push({ type: "Announcement 2", prompt: announcement2Prompt });
    prompts.push({ type: "Announcement 3", prompt: announcement3Prompt });
  
    Logger.log('🎉 Total prompts generated: ' + prompts.length);
    Logger.log('Prompt types: ' + JSON.stringify(prompts.map(p => p.type)));
  
    return prompts;
  }