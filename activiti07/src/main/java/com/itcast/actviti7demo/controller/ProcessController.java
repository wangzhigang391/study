package com.itcast.actviti7demo.controller;

import org.activiti.api.process.runtime.ProcessRuntime;
import org.activiti.api.task.runtime.TaskRuntime;
import org.activiti.engine.RepositoryService;
import org.activiti.engine.repository.Deployment;
import org.activiti.engine.repository.ProcessDefinition;
import org.activiti.engine.repository.ProcessDefinitionQuery;
import org.apache.commons.io.IOUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.multipart.MultipartFile;

import javax.servlet.ServletOutputStream;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;

/**
 * 处理流程相关的类
 */
@Controller
@RequestMapping("/pages")
public class ProcessController {
    //注入相关的流程处理的runtime,service
    @Autowired
    private ProcessRuntime processRuntime;
    @Autowired
    private TaskRuntime taskRuntime;
    @Autowired
    private RepositoryService repositoryService;

    @PostMapping("/deployment")
    public String deployment(@RequestParam("bpmn") MultipartFile file, HttpServletResponse response) throws IOException{
        //1.得到上传文件的文件名
        String fileName = file.getOriginalFilename();
        //2.实现流程部署
        Deployment deployment = repositoryService.createDeployment()
                .addBytes(fileName, file.getBytes()).deploy();
        //3.输出部署的ID
        String deploymentID = deployment.getId();
        System.out.println("======================================="+deploymentID);
        //查看token取值
        //DefaultCsrfToken csrfToken = (DefaultCsrfToken)request.getAttribute("_csrf");
        //System.out.println("--------------------"+csrfToken.getToken());
        return "ablank";
    }

    @RequestMapping("/search-Process")
    public String search(Model model){
        //1.查询流程定义信息
        //得到流程定义的查询器
        ProcessDefinitionQuery processDefinitionQuery = repositoryService.createProcessDefinitionQuery();

        List<ProcessDefinition> list = processDefinitionQuery.list();

        //2.将查询结果放入model中
        model.addAttribute("list",list);

        System.out.println("================="+list.size());
        //3.跳转页面
        return "searchProcess";
    }

    @RequestMapping("/viewBpmn")
    public void view(String processDefinitionId, String resourceType, HttpServletResponse response) throws Exception {
        //1.得到ProcessDefinitionQuery对象
        ProcessDefinitionQuery processDefinitionQuery = repositoryService.createProcessDefinitionQuery();
        //2.根据processDefinitionId，查询得到具体的一个ProcessDefinition对象
        ProcessDefinition processDefinition = processDefinitionQuery.processDefinitionId(processDefinitionId).singleResult();
        //3.通过processDefinition对象，得到deploymentID
        String deploymentId = processDefinition.getDeploymentId();
        //4.获取资源文件的名称
        InputStream is =null;
        if("bpmn".equals(resourceType)){
            String resourceName=processDefinition.getResourceName();//得到资源文件名称
            //5.获取bpmn文件输入流
             is = repositoryService.getResourceAsStream(deploymentId,resourceName);
        }

        //6.由response对象，得到outputStream
        ServletOutputStream outputStream = response.getOutputStream();
        //7.实现文件复制
        IOUtils.copy(is,outputStream);
        //8.关闭流对象
        outputStream.close();
        is.close();
    }

    @RequestMapping("/deleteDeployment")
    public String deleteDeployment(String deploymentId){
        //1.调用repositoryService的方法，实现删除操作
        repositoryService.deleteDeployment(deploymentId,true);

        //2.跳页面
        return "forward:search-Process";
    }

}
